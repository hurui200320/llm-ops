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
