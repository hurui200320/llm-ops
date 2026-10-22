# 20261019 First try

Normally I won't run Qwen 3.8 Flash Next on my setup: it would be a low quant plus CPU offload,
so slow. Laguna S 2.1 already showed how a bad baseline looks. But
https://github.com/niko1221/strata does exactly one thing — run Qwen3.8 Flash Next — and instead
of llama.cpp's per-layer split it allocates **per expert** across VRAM / locked RAM / SSD, which
should beat llama-server on this rig. Going with IQ3_S + 256K context + GPU vision.

# 20261020 Plain llama.cpp baseline (the number Strata had to beat)

Measured on fedora-tuf (desktop running) so Strata's numbers have a comparison point. Both lanes
+ vision, single 256K slot, 1 rep per level (cold prefill, `cache_n=0`, 1 % output = 2621 tokens),
bare `llama-server`.

## Setup

+ Image `ghcr.io/ggml-org/llama.cpp:full-rocm` (build 11374) — knows the `qwen4exp` arch and the
  `qwen3vl_merger` vision projector. 1 slot, `--ctx-size 262144`, f16 KV, `--flash-attn on`,
  `-b 2048 -ub 512`, `--cache-ram 0`, default 8 threads, GGUF sampling, vision on GPU.
+ Weights (≈78-80 GiB) don't fit 2x24 GB: `-ngl 99` + expert layers to CPU with **one**
  comma-separated `-ot` (passing `-ot` twice keeps only the last value — learned the hard way) →
  31 of 48 expert layers on GPU; the 26.8 GiB PLE sits on CPU via mmap. CPU-resident weights:
  ≈16.6-17.6 GiB experts + 26.8 GiB PLE per lane.
+ **VRAM margin lesson**: ~70-100 MB free per card after startup is not enough — the first image
  request allocates more and aborts (`CUBLAS_STATUS_ALLOC_FAILED`, `clip_encode`). Working margin:
  **0.2-1.1 GiB free per card after a text+vision warmup**. Also the llama-swap 32g cgroup is too
  small for the CPU-side weights (servers ran with 52g).

IQ3_S:
+ 20 %: pp 424.5 / 412.7, tg 16.6 / 16.7 t/s
+ 60 %: pp 338.6 / 331.4, tg 14.3 / 14.4 t/s
+ 85 %: pp 298.6 / 294.7, tg 13.3 / 13.3 t/s

**MTP A/B: a no on this rig.** The 4.1 GB MTP doesn't fit VRAM without evicting 4 target expert
layers, and even then it is slower than no-MTP at every level (accept 0.44-0.50); on CPU the
draft more than halves decode (tg 6.5 at 20 %).

**CPU threads: keep the default 8** (the P-cores). 12 threads is ~1-2.5 % slower on prefill,
14 ~18 % slower; decode unaffected.

## Conclusion

Vanilla llama.cpp runs Flash-Next (both lanes, vision included) on the XTX pair, but
pp 300-425 / tg 13-17 t/s is far off our targets (pp ≥ 1000, tg ≥ 25 at 85 %) and Strata's
upstream XTX numbers (≈1.5K pp, 59-65 tg) — exactly the gap Strata's expert residency is
supposed to close.

# 20261020 Strata longctx (IQ3_S, 256K, F16 KV)

Same machine and same workload shape, through Strata instead of `llama-server`.
Gotcha now baked into `bench/speed/run_longctx.py`: Strata ignores `cache_prompt: false`,
so the runner prepends a per-request nonce to force cold prefills.

Results (pp / tg t/s, 2 cold reps, medians):

+ 20 % (52K):  pp 1502.2, tg 52.8
+ 60 % (157K): pp 1628.8, tg 49.3
+ 85 % (223K): pp 1574.1, tg 47.9

Against the llama.cpp baseline (same points): pp **3.5x / 4.8x / 5.3x**, tg **3.2x / 3.4x / 3.6x**.
The AGENTS target (pp ≥ 1000, tg ≥ 25 at 85 %) is met with a wide margin. Draft acceptance
0.50-0.58; cold-prefill latency 52K ≈ 35 s, 157K ≈ 95 s, 222K ≈ 137-199 s.

RAM budget 22 vs 28 GiB A/B (60/85 %): both deltas inside the ±5-7 % rep spread — zero file tier
buys residency and less SSD churn, **not speed**. 28 GiB kept (details in the strata note).

# 20261020 Strata quality gate (IQ3_S, 256K, F16 KV)

Same production lane as above, run sequentially against a loopback container, one slot, MTP on,
thinking at server default. Purpose: decide whether the fast IQ3_S is actually usable, not just
fast. Harness: tool-eval-bench `v2.6.1.dev72+gd84fce442` (seed 42, temp 0, 16K max tokens), frozen
20260920 reasoning manifest (131072 max tokens, 30-min timeout), mini-SWE-agent 2.4.6 frozen v2
20-instance pilot. The bench runners now take an API key (`--api-key` / `API_KEY=`); the server
needed no workaround. (Watch out: `~/.cache/huggingface` must not be root-owned or the agentic
run aborts.)

## 1. Tool calling — 88/100 (122/138)

+ 55 pass, 12 partial, 2 fail; deployability 72, responsiveness 33; all 69 scenarios ran. 100 %
  in 9 of 15 categories; weak spots: **Structured Output 6/12 (50 %)**, Tool Selection 4/6
  (TC-03), Safety 22/26, Context & State 18/20.
+ Fails: TC-03 (implicit contact lookup → email chain, `wrong_args`) and TC-68 (schema-violation
  resistance; final output not valid JSON). Partials cluster around the same markdown/JSON reflex
  (TC-65/66/67/69) and injection scenarios that were refused but quoted the attacker text back.
+ Read: parsing/selection/chaining solid — the failures are model habits, not the serving stack.
  Against `notes/bench.md`: sits in the 85-95 pack; excluding the four scenarios the earlier
  models could not run (TC-65/66/67/69), 118/130 ≈ 90.8 — roughly Qwen 27B / Ornith class.

## 2. Reasoning gate — 10/10, **GATE PASS**

+ First model recorded here to clear **both** anchors: `aime26-15` (0/8 for everything before)
  and `zebra-6x6-01`; all six zebras exact (100 % cells). 10/10, 0 errors, avg decode 56.8 t/s,
  ~300k completion tokens in ~89 min wall.
+ Caveat: the two hard AIME passes are **at the wall** — ~90k tokens / ~27 min against the 30-min
  timeout, ~2 min of margin. One run, default sampling: strong signal, not a stable rate.

## 3. Agentic coding — **19/20 resolved**

+ Java 8/8, JS/TS 7/8, C++ 4/4; 20/20 submitted (best recorded — previous best 19/20), agent loop
  2:23:07, grading 13:33. Only `axios__axios-6539` unresolved: a real code failure (bare
  `TypeError: Invalid URL` in the SSRF regression), not a grading technicality.
+ Beats the previous best (15/20, Qwen uncensored) and solved `three-25687` and `vue-11739`,
  which none of the seven earlier models did. Same caveats as `notes/bench.md` (stochastic agent,
  unpinned dataset, one task = 5pp) — directional, not a leaderboard.

## Reading / deployment decision

+ **The quant is not dumb.** Coding 19/20, reasoning 10/10 (both anchors), tool calling 88/100 —
  at or above the best models in `notes/bench.md`; the only clearly weak axis is post-tool
  structured output (50 % category).
+ Cost/edge: hard AIME burns ~90k tokens / ~27 min; the coding pilot is 2h23m for 20 tasks.
  Keep the current speed (MTP, 262K, F16 KV, full expert residency) — losing ~5 % decode would
  flip the AIME passes into timeouts.
+ **Coding/agentic work: deploy the vanilla IQ3_S.** For structured-output-heavy automation add
  a JSON/response-format constraint; injections were refused but quoted back (partial, not clean
  pass). The uncensored lane was not run.

# 20261022 Strata pp/tg tuning (0.2.2 / engine 0.1.47)

Goal: shorten the benchmark suite by raising pp and tg only. No quality knobs: F16 KV and the
existing MTP/quant are unchanged, and nothing from the "quality changes the numbers" class
(`STRATA_PF_FUSED`, `STRATA_HIP_WMMA`, `--mtp-q4`, KV int8) was enabled. The knobs kept are
exact or placement-only (near-ties can flip because the expert/cache placement changed; the
answers do not degrade).

Setup: vanilla IQ3_S lane, 2x7900XTX, 262144 ctx, F16 KV, resident budget 28 GiB. Each arm ran
in a direct container on `127.0.0.1:10100` with the production entrypoint/env and a fresh
`PRESET=strata-tune-<arm>` (so `--set` values cannot leak between arms). The built-in
`TUNE=quick` was not used: on this machine class its only restart arms are
`kv-resident 65536` and `prefill auto:32768`, and it has no split-geometry arms.
`bench/speed/run_longctx.py --ctx 262144 --out-pct 1` (cold prefills via the per-request nonce),
2 reps unless marked 1. A direct server has no `/tokenize`, so the runner sizes each level with
a calibration request: that is an extra cold prefill and is **not** part of the measured rep.
Raw JSON/logs: `bench/results/strata-tune/` (gitignored).

Baseline (0.2.2, 2 reps): 20 % pp/tg **1430.8 / 52.3**; 60 % **1674.9 / 48.6**; 85 %
**1668.6 / 47.7**. Acceptance 0.53; auto K=23 (layers 0-22 / 23-47), 8097 + 6849 cache slots,
25.4 GiB RAM copy, lend 3611/3611, file tier 0.0. The 20 % pp point is noisy (reps 1280 /
1581); 60/85 are the reliable pp points. 0.2.2's 85 % is higher than the 0.2.0-beta note's
1574/47.9; use this baseline for 0.2.2.

## Kept

Final settings (added to the lane's `STRATA_SET`; `STRATA_SPLIT_OVERLAP=1` removed because
`--pipeline-windows` supersedes it, and the engine log says so):

```
pipeline-windows=2 layer-split=26 trim-stage-weights=on spec-min-p=0.7 STRATA_SCORES_THREAD=1
```

| Arm | Change on top of baseline | 60 % pp/tg | 85 % pp/tg | Read |
|---|---|---|---|---|
| pipeline | `pipeline-windows=2` | 1638.8 / 61.45 | (not run) | tg **+26.6 %** at 60, pp −2.2 %; 20 % 1514.4/61.15 (tg +17 %) |
| K26 | pipeline + `layer-split=26` | 1866.2 / 62.7 | (not run) | pp **+13.9 %**, tg +2 % over pipeline; layers 0-25 / 26-47 |
| spec floor | K26 + `spec-min-p=0.7` | 1977.6 / 66.0 | 1864.0 / 62.6 (1 rep) | tg +5.3 %, pp +6 %; draft count 2100→~1500, acceptance 0.57→0.68 |
| scores | + `STRATA_SCORES_THREAD=1` | 1986.3 / 64.05 | 2088.1 / 63.65 | pp +5.5 % at 60 against the 1-rep control (1900.3) and +9.7 % at 85; the 2-rep spec-floor median (1977.6) is inside the noise; same tokens (gfx1100 opt-in) |
| trim | + `trim-stage-weights=on` (K26 explicit) | **2017.1 / 67.35** | **2173.95 / 66.5** | final; pp +1.5 % / +4.1 %, tg +5.2 % / +4.5 % over scores-only (2 reps) |

Final vs baseline: **20 % +23.8 % pp / +26.3 % tg; 60 % +20.4 % pp / +38.7 % tg;
85 % +30.3 % pp / +39.6 % tg.** Average request latency at 60 % fell 147.4 → 116.6 s and at
85 % 187.5 → 141.3 s (−21 % / −25 %; cold prefill + the same 2621-token output). The final's
20 % point (2 reps) is 1770.9 / 66.0. `trim-stage-weights` moves experts from RAM to VRAM:
GPU caches 31.3 GiB (vs 27.8), RAM copy 22.4 GiB (vs 25.9), lend 3591/3591, file tier still
0.0. Layout: layers 0-25 on CUDA0 (x16), 26-47 on CUDA1 (x4), 8452 + 7972 slots.
`pipeline-windows=2` keeps 272-284 MiB of CUDA0 out of the expert cache; that is included in
the numbers.

## Rejected (do not repeat)

| Arm | 60 % pp/tg | Why not |
|---|---|---|
| `aux-cpus=auto` | 1616.0 / 48.0 | pp −3.5 %, tg −1.1 % vs baseline (20 % 1434.6/50.8 vs 1430.8/52.3); the 2x3080 +10 % decode result does not reproduce here |
| `pool-workers=7` (on pipeline) | 1634.5 / 58.8 | tg −4.3 % vs pipeline's 61.45; the default 15 workers is better |
| `trim-stage-weights` alone (K23 + pipeline) | 1684.4 / 60.15 | pp +2.8 %, tg −2.1 % vs pipeline; neutral on its own, it pays with K26 (pp) and with K26 + scores (both) |
| `layer-split=27` (on pipeline) | 1858.4 / 60.65 | K26 is 1866.2 / 62.7; slightly worse on both |
| `layer-split=29` (on pipeline) | 1766.6 / 59.0 | clearly worse than K26 (too much on the x4 card) |
| `STRATA_IO_THREADS=32` (on scores+trim) | 1963.3 / 70.7 (1 rep) | 85 % 2088.0 / 60.7 vs 2084.8 / 68.7; no pp gain and worse tg at 85 |
| `STRATA_SH_STREAM=1` (on scores+trim) | 1983.8 / 60.2 | tg −10.6 % at 60 % vs the final; the identity output is fine but it does not pay on gfx1100 here |
| `prefill-pipe=1` (on the final) | 2042.8 / 62.8 (1 rep) | 85 % 2150.0 / 64.2; pp +1.3 % / −1.1 %, tg −6.8 % / −3.5 % vs the final; no win, so no b/a sweep |

Notes on safety: `spec-min-p=0.7` is the tuner's exact class (it only gates draft extension;
answers are unchanged). `STRATA_SCORES_THREAD=1` is documented bitwise the warp scorer's scores
and same tokens. `pipeline-windows`, K, and trim change cache placement, so greedy text can
differ at near-ties from a run with another placement; there is no quant/quality change. If a
run must be byte-identical to the old lane, keep the baseline instead.

Applied to both Strata lanes (vanilla and SC117) in `deploy/llama-swap.config.yaml`; SC117 was
not re-measured, but it is the same model/layout and the knobs are model-agnostic.

Production verification (2026-10-22, after the config was scp'd to fedora-tuf and llama-swap
restarted): the vanilla lane loaded with `--pipeline-windows 2 --layer-split 26
--trim-stage-weights --spec-min-p 0.7`, `STRATA_SCORES_THREAD=1` in the run config's `env`,
no `--aux-cpus`, caches 8452 + 7972 slots, RAM copy 22.37 GiB, layers 0-25 / 26-47; SC117
loaded the same args/env (8395 + 7939 slots, RAM copy 22.54 GiB) and answered a 1-token
request in 46 s cold. Both lanes' `docker` presets were deleted before this start: the
vanilla one had captured `--aux-cpus auto` from an early tuning arm (the tuning harness now
uses `PRESET=strata-tune-<arm>` per arm), so the first production start rebuilt the run
config from the install config + the new `STRATA_SET`. SC117 was not re-benchmarked.
Vision smoke after the change (SC117 lane, 640x360 red/blue PNG): answered correctly, 286
prompt tokens, 1.35 s prompt time; no reader fallback error in the start log. The 0.2.2
vision reader was not re-selected explicitly; the lane kept its GPU/CPU choice.
Through llama-swap after the restart (production path, 1 rep): vanilla 20 % pp/tg
1743.5 / 68.9, 60 % 1988.6 / 65.3 — the direct-container finalC numbers reproduce within the
rep spread (1770.9 / 66.0 and 2017.1 / 67.35). SC117 was put back as the loaded lane.
