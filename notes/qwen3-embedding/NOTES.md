# 20260917

llama.cpp cannot change the output size, so use vLLM:

```bash
docker run --name qwen3-embedding-8b --init --rm \
    --device /dev/dri \
    --device /dev/kfd \
    --group-add=video \
    --group-add=render \
    --security-opt label=disable \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    -p 8000:8000 \
    --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai-rocm:latest \
    --model alexliap/Qwen3-Embedding-8B-FP8-DYNAMIC \
    --runner pooling --convert embed \
    --quantization compressed-tensors \
    --hf-overrides '{"is_matryoshka": true}'
```

Using dual AMD GPUs is still broken.

Compressed (FP8) is only supported on MI300+, not the 7900 XTX. Giving up.

# 20261120

Try again as I need embedding for work. Trying the following command:

```bash
docker run --name qwen3-embedding-8b --init --rm \
    --device /dev/dri \
    --device /dev/kfd \
    --group-add=video \
    --security-opt label=disable \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    -p 8081:8000 \
    --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai-rocm:latest \
    --model Qwen/Qwen3-Embedding-8B \
    --runner pooling --dtype float16 \
    --max-model-len 32768 \
    --hf-overrides '{"is_matryoshka": true}'
```

Failed with:

```text
ValueError: To serve at least one request with the model's max seq len (32768),
(4.5 GiB KV cache is needed, which is larger than the available KV cache memory
(0.48 GiB). Based on the available memory, the estimated maximum model length is 3520.
```

## Resolution

The weights are not the problem - `Qwen/Qwen3-Embedding-8B` loads in **14.95 GiB**
fp16. The default `--gpu-memory-utilization 0.9` (22.79 GiB of 23.98 GiB) minus
0.81 GiB peak activation and the PIECEWISE CUDA-graph reservation left only
**0.48 GiB** for KV, while 32k needs 4.5 GiB.

Working single-GPU command (both GPUs idle, serves on `:8081`):

```bash
docker run -d --name qwen3-embedding-8b \
    --device /dev/dri \
    --device /dev/kfd \
    --group-add=video \
    --security-opt label=disable \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    -p 8081:8000 \
    --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai-rocm:latest \
    --model Qwen/Qwen3-Embedding-8B \
    --runner pooling --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --max-num-batched-tokens 8192 \
    --hf-overrides '{"is_matryoshka": true}'
```

Measurements (vLLM 0.34.0, ROCm 7.15/Gfx1100):

| metric | value |
|---|---|
| model loading | 14.95 GiB, 5.9 s |
| free VRAM on startup | 23.68 / 23.98 GiB |
| peak activation | 0.81 GiB |
| CUDAGraph memory | 0.0 GiB |
| available KV cache | 6.95 GiB |
| GPU KV cache size | 34,816 tokens |
| max concurrency @ 32,768 | **1.06x** |

Embeddings are bit-identical (cosine 1.0) to a reference request from the same
config; output dim 4096, L2 norm 1.0.

`--enforce-eager` costs nothing here: pooling models never decode, and vLLM already
downgrades them to PIECEWISE graphs, so dropping capture entirely only frees memory.
`--max-num-batched-tokens 8192` caps the prefill activation peak. Per request, 32k
allows one request at a time; shorter inputs scale (4x concurrent at 8k).

## No pipeline parallelism either

`--pipeline-parallel-size 2 --tensor-parallel-size 1` is architecturally the right
multi-GPU shape for this link: PP is p2p only, no all-reduce, and one 32k handoff is
32768 x 4096 x 2 B = 268 MiB ~= 77 ms at 3.5 GB/s versus ~20 s of prefill, i.e. under
0.5%. It nevertheless cannot start: vLLM builds an RCCL communicator per worker and
runs a warm-up `ncclAllReduce` in `PyNcclCommunicator._init_comm` *before* loading
weights, which dies on this PCIe topology:

```text
RuntimeError: NCCL error: unhandled cuda error
HIP failure 'the operation cannot be performed in the present state'
  at rccl/build/hipify/src/enqueue.cc:2119
```

Directly probing the only op PP needs (`torch.distributed` send/recv, 2 ranks, 1 KB
tensor) fails at the first `irecv` with the same error, with `NCCL_P2P_DISABLE=1`
and unset, and with `HSA_ENABLE_SDMA=0`. See [`../rccl.md`](../rccl.md) for the flag
analysis; the blocker is not transport selection, any RCCL op on this pair fails at
enqueue.

For 2x throughput, run **two independent TP=1 replicas**, one per GPU, instead: no
RCCL, no pipeline bubble, full 32k each. The display card only leaves ~1.2 GiB for
the desktop at 0.95 utilization, so it is the awkward slot.

## Weight/ KV quantization

- FP8 weight quant is the same wall as 20260917: gfx1100 has no fp8 WMMA, it needs
  gfx94x (`torch._scaled_mm`).
- `--kv-cache-dtype fp8` would halve KV to 2.25 GiB and is the only in-engine lever
  left, but it is a last resort per repo policy.
- If more per-instance KV is needed, prefer an AWQ/GPTQ int4 checkpoint over KV quant,
  weighing the embedding-quality cost.
