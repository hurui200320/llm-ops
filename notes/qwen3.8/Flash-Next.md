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
