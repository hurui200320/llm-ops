# 20261004 RCCL requirement, hardware evidence, and engine verdict

**Use llama.cpp `--split-mode layer` (current split 35/31).** GPU1 sits behind
the Z790 PCH, which cannot route PCIe atomics: RCCL-based multi-GPU engines
are blocked. llama.cpp's internal tensor-mode fallback works without RCCL,
but its allreduce traffic makes it slower on this link.

## RCCL requirement and hardware evidence

[ROCm issue #6074](https://github.com/ROCm/legacy-rocm-build/issues/6074),
closed by `harkgill-amd` on 2026-09-29:
> PCIe atomics will continue to be a dependency for RCCL going forward.

[AMD docs](https://rocm.docs.amd.com/en/latest/conceptual/pcie-atomics.html):
upstream bridges must support **and route** atomics. This PCH/DMI path cannot;
no BIOS setting or driver update fixes the hardware limitation.

Evidence on fedora-tuf (20261004):
```text
$ dmesg | grep -i "PCIE atomic"
amdgpu 0000:09:00.0: PCIE atomic ops is not supported
```

| device | path | AtomicOpsCap | AtomicOpsCtl |
|---|---|---|---|
| `03:00.0` GPU0 | CPU root port | `32bit+ 64bit+` | `ReqEn+` |
| `09:00.0` GPU1 | PCH via `00:1b.4` | `32bit+ 64bit+` | `ReqEn-` |
| `00:1b.4` Z790 PCH port | GPU1 upstream | **`Routing- 32bit- 64bit-`** | `ReqEn- EgressBlck-` |

GPU1 supports atomics, but its upstream port does not. Its x4 Gen4 CPU path
is also bandwidth-limited (~6 GB/s ceiling, ~3.5 GB/s practical).

Thread findings:
- **Not vLLM-specific:** minimal `torch.distributed` and standalone HIP
  `ncclCommInitAll`/`ncclAllReduce` repros initialize successfully, then fault
  on the first collective.
- **Regression:** RCCL 2.27.7 / ROCm 7.2.1 failed on 2x 7900 XTX with one
  PCH-attached card; ROCm 7.2.0 worked. JartX's 7.2.0-amdclang rebuild worked;
  cadamcat traced RCCL PR #2124's removal of `NDEBUG` to device asserts pulling
  hostcall buffers into Generic kernels.
- **Atomics are not sufficient:** miversen33 (2026-07), EPYC 7601 + 3x 7900 XTX,
  all atomics enabled, RCCL 2.30.4 / ROCm 7.14: first-collective nil-address /
  gfxhub TCP fault. AMD requested a separate issue; unresolved as of this note.
  Thread operators report llama.cpp's coarse layer split as more forgiving.

## Engine verdict for 27B + 256k f16 KV on 2x24 GB

| engine | verdict / blockers |
|---|---|
| vLLM | **Out:** TP≥2 requires RCCL; TP=1 exceeds 24 GB. Official FP8 checkpoint is 38 GB before KV, leaving only tens of thousands of tokens on 2x24 GB. gfx1100 lacks native FP8; Qwen GDN MTP needs source builds/local patches (vllm #43559; shisa-ai/hipEngine `docs/VLLM_RDNA3.md`). |
| SGLang | **Out:** TP>1 uses torch-dist/RCCL. Radeon support is initial (ROCm 7.14+, `SGLANG_USE_AITER=false` workarounds); Qwen3.8-27B recipe is NVIDIA-only/single-GPU. |
| TGI | Out: tested AMD support covers Instinct MI210/MI250/MI300, not consumer Radeon. |
| LMDeploy / MLC-LLM / exo / Ollama / LM Studio / TabbyAPI / KTransformers | Out: CUDA-only, llama.cpp wrappers, or missing MTP / 256k / vision parity. |
| llama.cpp | **Use layer split:** direct P2P / host-staged copies, no RCCL dependency. |

# 20261005 llama.cpp split modes

Verified against running `ghcr.io/ggml-org/llama.cpp:full-rocm`,
b11096/c550d2f60:
- `none`: model + KV cannot fit one 24 GB card.
- `row`: fails with `device ROCm0 does not support split buffers`.
  Split-buffer support exists only in SYCL/Hexagon in this tree, not CUDA/HIP.
- `tensor` (experimental): accepts `qwen35`, **works without RCCL, but slower**.
  `GGML_HIP_RCCL=OFF`; `libggml-hip.so` has no RCCL symbols. The communication
  chain (NCCL/RCCL → internal → butterfly) selects internal AllReduce
  (`ggml/src/ggml-cuda/allreduce.cu`): D2H pinned staging → peer spin-wait → H2D.
  No atomics/P2P required, but each row-split output crosses PCIe twice.

Estimated GPU1-link traffic from the GGUF (`qwen35`, 64 main + 1 MTP block,
hidden 5120, Q8_0 29 GB), at ~3.5 GB/s:

| mode | TG, b=1 | PP, ub=256 |
|---|---|---|
| `layer`: 5120 f32 residual, both ways | 40 KB ≈ 12 µs (0.02% of 50 ms/token) | 10.5 MB, hidden by ~0.38 s/ubatch pipeline |
| `tensor`: row-split outputs, D2H+H2D per GPU | ~38.5 MB ≈ **11 ms/token**, plus 350–400 handshakes | ~8 GB ≈ **2.3 s**, vs ~0.2 s compute |

Tensor mode's reduced weight traffic is outweighed by communication; estimated
pp/tg slowdown is 2–4x. An OpenJev (Qwen 3.8 27B derivative) prompt-only run
confirmed slower **pp** than layer mode; tg was not measured. Keep layer split;
the baseline below already includes its negligible PCH handoff cost.

# 20261120 Pipeline parallelism is also blocked: RCCL p2p fails, not just collectives

The 20261004 verdict was based on collectives (all-reduce). vLLM's
`--pipeline-parallel-size 2 --tensor-parallel-size 1` needs only point-to-point, so it
was worth re-testing as an escape hatch. It is blocked too.

**vLLM startup** (vLLM 0.34.0, `vllm/vllm-openai-rocm:latest`, Qwen3-Embedding-8B,
PP=2/TP=1, 32768 max len): RCCL's `ncclCommInitRank` and vLLM's process groups
initialize fine; the failure is the one-element warm-up all-reduce in
`PyNcclCommunicator._init_comm`, which runs before any weights are loaded:

```text
RuntimeError: NCCL error: unhandled cuda error
Last error:
HIP failure 'the operation cannot be performed in the present state'
  at /__w/rockrel/rockrel/build/comm-libs/rccl/build/hipify/src/enqueue.cc:2119
```

**Minimal probe** (`torch.distributed`, 2 ranks, 1 KB tensor, `send`/`recv` only -
the sole op PP needs): `init_process_group('nccl')` succeeds on both ranks, then the
first `irecv` enqueue fails with the identical `enqueue.cc:2119` error.

| flags | `init_process_group` | first `send`/`recv` |
|---|---|---|
| none | OK | **fail** `enqueue.cc:2119` |
| `NCCL_P2P_DISABLE=1` | OK | **fail** `enqueue.cc:2119` |
| `NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0` | OK | **fail** `enqueue.cc:2119` |

`NCCL_P2P_DISABLE` is honored - RCCL logs `NCCL_P2P_DISABLE set by environment to 1`
and still builds `Ring 0 : 1 -> 0 -> 1 ... type PHB/PIX, nChannels 1`. Disabling P2P
only changes the chosen transport; the enqueue still faults, so the SHM fallback is
not a workaround here.

Related RCCL warning on this host, unrelated to the atomics routing limitation:
`Missing "iommu=pt" from kernel command line which can lead to system instability or
hang!` (and `DMA_BUF Support Enabled ... pci_p2pdma`).

Conclusions:

- No RCCL-based engine is viable on this pair, at any parallelism degree. This covers
  vLLM PP as well as TP, and any torch-distributed path.
- For stateless workloads (embeddings), run **N independent TP=1 replicas** instead of
  one PP/TP job - no RCCL, no pipeline bubble, and each replica keeps full per-GPU
  memory. Do not spend more time on RCCL flags.
- `HSA_ENABLE_SDMA=0` is a workaround for genuine SDMA ring hangs, not for atomics or
  this enqueue fault. The SDMA timeouts in `hardware-failure.md` were a symptom of an
  already-removed device, not a cause.
