# 20261003 First time deploy

There are a lot of decision models. OpenJev is based on Qwen3.8 27B.
So far llama docker image doesn't have this release yet, so we have to build our own image.

The decision model is based on Qwen 3.8 27B, so it can generate text, but not recommended.
The LLM can support 256k context, but the decision fine tune is only trained and validated
for 16k context. Longer than this may degrade the performance.

The total context is 128k, divided into 8 private 16k slots (`--no-kv-unified`).

# 20261003 Latency optimization

**Use layer/256 plus `--no-cache-idle-slots` as the latency-first default for
short decisions.** Use layer/512 with the same cache flag when 4K latency is the
priority. Retain the rest of the existing OpenJev command:

```text
-b 2048 -ub 256
--main-gpu 0 --split-mode layer --tensor-split 32,33
--no-cache-idle-slots
```

## Setup and confirmed results

Runner: two RX 7900 XTX (24 GiB each), i7-13700K, llama.cpp `b11374-b92761a51`.
Fixed image (no pull/rebuild):
`sha256:8a19ec0d442120f8efcec4ceadead0433e8f8bc58cd5aae71882cd5e5f6a5094`.
`bench/systemone-latency/run_latency.py` tested llama-swap at
`http://127.0.0.1:8080` over a persistent connection, one request at a time.
Each confirmation used **200 fresh requests per workload**, 10 excluded warmups,
seed 42, and one choice question with 16 described options. Timings include HTTP
and full response transfer, excluding loading, calibration, parsing, and payload
construction. **Text only, with vision loaded**; no screenshot or concurrency test.

Fixed settings: `-b 2048`, tensor proportions `32,33`, main GPU 0, all layers
offloaded, 131072 context across 8 private 16384-token slots, F16 KV, flash
attention, GPU vision projector, image budget 1024–8192, direct-I/O loading,
`--metrics`, 4096 MiB saved prompts, 16 checkpoints, minimum step 8192.

| Split / ub | Idle-slot caching | 1K p50 / p95 ms | 4K p50 / p95 ms | Input p50 (1K / 4K) |
| --- | --- | ---: | ---: | ---: |
| tensor / 1024 | original | 1243.09 / 1274.89 | 3960.55 / 4052.50 | 1042 / 4136 |
| layer / 256 | disabled | **779.98 / 781.27** | 2514.65 / 2523.76 | 1043 / 4138 |
| layer / 512 | disabled | 812.56 / 817.12 | **2443.15 / 2453.40** | 1026 / 4135 |

All workloads returned 200/200 valid, expected decisions with no errors and
**zero cached prompt tokens**. Layer/256 improves baseline medians by **37.3%
at 1K and 36.5% at 4K**. Against layer/512, it is about 33 ms (4.0%) faster at
1K; layer/512 is 72 ms (2.8%) faster at 4K. Calibration slightly changes prompt
sizes, so choose using representative traffic, not a fixed workload crossover.

## Experiment takeaways and limits

- The initial 12-way sweep tested layer/tensor/row with ub 256/512/1024/2048
  (50 requests per workload). Layer won; bigger ub was not universally faster,
  with layer degrading above 512. A 200-request repeat confirmed the tradeoff.
  Layer/256 and /512 used about 18.56/18.66 GiB per GPU after testing (not peak).
- Row failed to load: `device ROCm0 does not support split buffers`. This build
  does not support it; this was not a measured OOM or slow result.
- Disabling eager idle-slot caching cut layer/256's fresh 1K median by about
  16.6% in screening. Prefer it over `--cache-ram 0`: it retains the saved-prompt
  budget, while the latter's smaller 1K prompt confounded its extra gain.
  Identical-request reuse also reported zero cached tokens, **not a prefill-cache
  hit**. Edited histories and concurrent sessions remain untested. Checkpoints
  are completion-only in the reviewed upstream code, so disabling them is not
  an obvious optimization here.
- Neither candidate reaches the initial 50 ms goal. The
  [upstream 43 ms median](https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp)
  on one RTX PRO 6000 is not like-for-like: prompt sizes, option descriptions,
  cache state, and quantization are unspecified.
- Next levers: trim irrelevant state, batch independent questions sharing a
  state/prefix, and keep the model loaded. Benchmark synthetic screenshots before
  changing image budgets; smaller weights/models need separate accuracy checks.
  F16 KV stayed unchanged throughout.
