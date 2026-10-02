# 20261002 Model startup and switching

Selected for all seven models: **`--load-mode dio --fit off` and a persistent
COMGR cache**. Broad seven-model rotation took **39.7% less startup time** than mmap;
mmap remains preferable for a small, fully cached subset. Inference settings are unchanged.

Runner: fedora-tuf, i7-13700K, 64 GB RAM, dual 7900 XTX, Samsung 980 PRO with
unencrypted XFS. GPU 0 is CPU-connected PCIe 4.0 x16; GPU 1 is chipset-connected
with an upstream x4 bottleneck. Tested llama.cpp `b11312-0c1e57098` and llama-swap
v256 (`6701d0d`); deployment keeps mutable `full-rocm` for manual updates, so
results apply to this build, not future pulls.

## Method

- Run isolated deployment-equivalent containers one at a time, with no models
  loaded in llama-swap. Retain 32 GiB/no-swap limits, context/slots, F16 target/draft
  KV, GPU placement, sampling, batch/ubatch, checkpoints, speculation and inference
  warmup. No CPU layer/expert offload or quantization changes.
- Time Docker launch to `/health` 200 (about 50-ms polling), then check a streamed
  request's first token. Excludes cache preparation, shutdown and llama-swap
  scheduling/one-second health polling. Production stops took 0.4-0.6 s, not 10 s.
- Cold: stop the container, evict only tested weight/projector/draft files with
  `POSIX_FADV_DONTNEED`, verify <1% residency using `fincore`. Warm: populate mappings
  and verify ≥99%; sequential buffered reads alone did not reliably warm this host.
  Never flush the whole host cache. Cached weights are not cached inference KV.
- Warm preparation is outside the serving cgroup: pages may remain charged to
  their original loader. Compare anonymous RAM, residency and storage reads, not
  cold/warm `memory.current` alone.

## Fitting and COMGR cache

Three warm repetitions per combination, rotating order; median ready time (s):

| Model | Fit on, disposable cache | Fit off, disposable cache | Fit on, persistent cache | Fit off, persistent cache |
| --- | ---: | ---: | ---: | ---: |
| Muse | 4.717 | 3.999 | 4.562 | 3.895 |
| Qwen official | 4.714 | 4.143 | 4.638 | 4.087 |

- **Keep `--fit off`:** saves ~0.6-0.7 s by skipping memory estimation when placement
  is already explicit. Qwen's fitting failure (`n_gpu_layers` already set) was not
  a model-load failure.
- **Keep persistent COMGR:** warmed compiler entries saved only ~0.05-0.15 s,
  less robust than the fitting result. Reuse was verified by unchanged entry
  timestamps; new workloads still compile. Startup entries used ~126 KiB, growing
  by ~20 MiB with vision checks.
- **Reject `prune_expiration=0h`:** copied from AMD docs, but this runtime reported
  `Unknown key: 'prune_expiration'` and silently disabled caching at normal verbosity.
  Verbose diagnostics exposed it; exclude those empty-cache runs.

Use a Docker named volume at `/root/.cache/comgr`, `AMD_COMGR_CACHE=1`,
`AMD_COMGR_CACHE_DIR=/root/.cache/comgr`, and this tested pruning policy:

```text
prune_interval=1h:cache_size=0%:cache_size_bytes=256m:cache_size_files=0
```

256 MiB is a periodic pruning target, not a hard quota. Normal operation does not
enable verbose COMGR diagnostics.

## Loading-mode comparison

- `mmap`: pre-populates mapped weights, then uploads; fastest with cached weights,
  but cold population can hit the container's file-cache memory limit.
- `none`: buffered reads with staged async GPU uploads where supported; avoids mmap
  population, but still uses file cache. A compromise, not the fastest endpoint.
- `dio`: direct reads with staged async GPU uploads; bypasses cache for most
  target/draft weights even when resident. Metadata/projector I/O can still be buffered.

Fit off and persistent COMGR throughout. Two repetitions per model/mode/cache
state (84 successful starts); median ready time (s), **cold / fully cached**:

| Model | mmap | none | dio |
| --- | ---: | ---: | ---: |
| Gemma 26B | 10.497 / 3.974 | 6.899 / 5.139 | 6.240 / 8.325 |
| Gemma 31B QAT | 6.571 / 3.185 | 5.410 / 3.958 | 4.806 / 5.435 |
| Muse | 7.889 / 3.876 | 6.133 / 4.693 | 5.611 / 5.406 |
| Ornith official | 14.160 / 4.950 | 8.136 / 6.533 | 7.248 / 7.357 |
| Ornith abliterated | 13.560 / 4.914 | 8.249 / 6.440 | 7.227 / 7.257 |
| Qwen official | 9.037 / 4.081 | 7.168 / 5.310 | 6.287 / 6.801 |
| Qwen uncensored | 9.087 / 4.082 | 7.186 / 5.304 | 6.292 / 6.469 |

`dio` wins cold for every model; `mmap` wins fully cached. Only two repetitions:
Gemma 26B cold mmap ranged 8.284-12.710 s, so these are tradeoffs, not universal ratios.

Ornith explains the cold mmap penalty: its ~36 GiB weight/projector set exceeds
the 32 GiB cap, causing reclaim/refaults and 41.57 / 67.36 GiB storage reads.
`dio` read ~35.23 GiB with a ~3.43 GiB startup cgroup peak and no cap/reclaim/refault
events. `none` avoided rereads but still hit the cap. Keep the serving memory limit.

### Broad model rotation: why choose dio

Four seven-model cycles per mode (two forward, two reverse), rotating strategy
order. Evict before each strategy, **no forced warming/eviction between requests**;
84 additional starts across the ~192 GiB collection.

| Mode | Median seven-model startup sum (s) |
| --- | ---: |
| mmap | 71.188 |
| none | 56.674 |
| dio | 42.957 |

`dio` wins every model and cuts the cycle by **39.7%** versus mmap, excluding
inference/stop/scheduler time. mmap starts had ≤~0.9% resident weights; `none`
retained up to ~55%, still insufficient to win. Direct I/O makes reads predictable
without displacing host cache with the collection. Use mmap instead if repeated
models genuinely stay fully cached.

## Validation and deployment

- First-token checks passed in all 168 loading/rotation starts. Fitting/cache
  savings were not shifted into first inference; loading-mode differences were
  mainly startup. All tests had zero container swap/OOM.
- Selected dio mode passed synthetic long-context checks: ~227-228K tokens for
  256Ki models, ~112.6K for Muse's 128Ki slots; both Gemma 26B slots and all four
  Muse slots populated. Cold prefill, cached appends, recent-history edits and
  128-token decode passed, preserving prefixes. Maximum sampled anonymous RAM
  across models was 17.683 GiB.
- Plain/streamed tool-call/result loops, strict JSON schema and synthetic red-image
  input passed for every model in both dio and reference mmap. After config sync,
  llama-swap validation/watch-config succeeded and all **28 proxy compatibility
  checks** passed. Effective startup settings and unchanged inference arguments
  were verified.
- Speeds broadly matched [prior serving benchmarks](bench.md#speed); this was not
  a same-prompt longctx speed A/B or quality benchmark, and did not repeat full
  checkpoint saturation, near-limit context or heavy-desktop tests. Qwen still
  briefly reached 23.952 GiB on GPU 1 (~33 MiB free), settling near 23.70 GiB:
  **startup tuning does not fix its VRAM-headroom issue**.

## References

- [Tested llama.cpp loading paths](https://github.com/ggml-org/llama.cpp/blob/0c1e57098/src/llama-model-loader.cpp)
- [Tested llama.cpp auto-fit entry point](https://github.com/ggml-org/llama.cpp/blob/0c1e57098/common/common.cpp)
- [COMGR cache documentation](https://github.com/ROCm/llvm-project/blob/amd-staging/amd/comgr/README.md)

Runtime verification takes precedence over mutable docs, especially COMGR policy syntax.
