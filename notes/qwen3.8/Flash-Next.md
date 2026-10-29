Current Qwen lane settings, deploy record and upgrade mechanics:
[`notes/strata.md`](../strata.md). This note is the model and tuning history.

# 20261019 First try

Normally I won't run Qwen 3.8 Flash Next on my setup: it would be a low quant plus CPU offload, so
slow. Laguna S 2.1 already showed how a bad baseline looks. But
https://github.com/niko1221/strata does exactly one thing — run Qwen3.8 Flash Next — and instead of
llama.cpp's per-layer split it allocates **per expert** across VRAM / locked RAM / SSD, which should
beat llama-server on this rig. Going with IQ3_S + 256K context + GPU vision.

# 20261020 Plain llama.cpp baseline (the number Strata had to beat)

Vision on GPU, single 256K slot, IQ3_S, f16 KV, `--flash-attn on`, `-b 2048 -ub 512`,
`--cache-ram 0`, default 8 threads. Layer split: `-ngl 99` + expert layers to CPU with one
comma-separated `-ot`, 31 of 48 expert layers on GPU; the 26.8 GiB PLE sits on CPU via mmap.
CPU-resident weights: ≈16.6-17.6 GiB experts + 26.8 GiB PLE per lane.

+ 20 %: pp 424.5 / 412.7, tg 16.6 / 16.7 t/s
+ 60 %: pp 338.6 / 331.4, tg 14.3 / 14.4 t/s
+ 85 %: pp 298.6 / 294.7, tg 13.3 / 13.3 t/s

MTP: a no on this rig. The 4.1 GB MTP doesn't fit VRAM without evicting 4 target expert layers, and
even then it is slower than no-MTP at every level (accept 0.44-0.50); on CPU the draft more than
halves decode (tg 6.5 at 20 %).

CPU threads: keep the default 8 (the P-cores). 12 threads is ~1-2.5 % slower on prefill, 14 ~18 %
slower; decode unaffected.

# 20261020 Strata longctx (IQ3_S, 256K, F16 KV)

Same machine and workload shape, through Strata instead of `llama-server`. Gotcha now baked into
`bench/speed/run_longctx.py`: Strata ignores `cache_prompt: false`, so the runner prepends a
per-request nonce to force cold prefills.

+ 20 % (52K):  pp 1502.2, tg 52.8
+ 60 % (157K): pp 1628.8, tg 49.3
+ 85 % (223K): pp 1574.1, tg 47.9

Against the llama.cpp baseline: pp **3.5x / 4.8x / 5.3x**, tg **3.2x / 3.4x / 3.6x**. Draft
acceptance 0.50-0.58.

# 20261020 Quality gate (IQ3_S, 256K, F16 KV)

Initial gate on the production lane: tool calling 88/100 (122/138; failures TC-03 and TC-68, with
partials around the markdown/JSON reflex in TC-65/66/67/69), reasoning 10/10 (the first model here
to clear both `aime26-15` and `zebra-6x6-01`; ~300k completion tokens in ~89 min wall, 56.8 t/s
average decode), and agentic coding 19/20. The quant is not dumb; the only clearly weak axis was
post-tool structured output (50 %). Caveat: the two hard AIME passes were at the wall (~90k tokens /
~27 min against the 30-min timeout), so this is a strong signal, not a stable rate.

The later full runs supersede this gate; all current cross-model results and per-scenario detail are
in [`notes/bench.md`](../bench.md). This note keeps the tuning and upgrade history.

# 20261022 Strata pp/tg tuning (0.2.2 / engine 0.1.47)

Goal: raising pp and tg speed. No quality knobs: F16 KV and the existing MTP/quant are unchanged,
and nothing from the "quality changes the numbers" class (`STRATA_PF_FUSED`, `STRATA_HIP_WMMA`,
`--mtp-q4`, KV int8) was enabled. The knobs kept are exact or placement-only (near-ties can flip
because the expert/cache placement changed; the answers do not degrade).

The built-in `TUNE=quick` was not used: on this machine class its only restart arms are
`kv-resident 65536` and `prefill auto:32768`, and it has no split-geometry arms.
`bench/speed/run_longctx.py --ctx 262144 --out-pct 1` (cold prefills via the per-request nonce),
2 reps unless marked 1. A direct server has no `/tokenize`, so the runner sizes each level with a
calibration request: that is an extra cold prefill and is **not** part of the measured rep.

Baseline (0.2.2, 2 reps):
+ 20 % pp/tg 1430.8 / 52.3
+ 60 % pp/tg 1674.9 / 48.6
+ 85 % pp/tg 1668.6 / 47.7

Acceptance 0.53; auto K=23 (layers 0-22 / 23-47), 8097 + 6849 cache slots, 25.4 GiB RAM copy,
lend 3611/3611, file tier 0.0. The 20 % pp point is noisy (reps 1280/1581); 60/85 are the reliable
pp points.

## Kept

Final settings (added to the lane's `STRATA_SET`; `STRATA_SPLIT_OVERLAP=1` removed because
`--pipeline-windows` supersedes it, and the engine log says so):

```
pipeline-windows=2 layer-split=26 trim-stage-weights=on spec-min-p=0.7 STRATA_SCORES_THREAD=1
```

| Step | 20 % pp/tg | 60 % pp/tg | 85 % pp/tg | Note |
| --- | --- | --- | --- | --- |
| baseline | 1430.8 / 52.3 | 1674.9 / 48.6 | 1668.6 / 47.7 | auto K=23 |
| `+ pipeline-windows=2` | 1514.4 / 61.15 | 1638.8 / 61.45 | — | pp -2.2 %, tg +26.6 % at 60 % |
| `+ layer-split=26` | — | 1866.2 / 62.7 | — | both improve |
| `+ spec-min-p=0.7` | — | 1977.6 / 66.0 | 1864.0 / 62.6 | acceptance 0.57 → 0.68 |
| `+ STRATA_SCORES_THREAD=1` | — | 1986.3 / 64.05 | 2088.1 / 63.65 | pp slightly improves |
| `+ trim-stage-weights=on` | — | 2017.1 / 67.35 | 2173.9 / 66.5 | pp slightly improves, tg in margin |

Final vs baseline: 20 % pp +23.8 %, tg +26.3 %; 60 % pp +20.4 %, tg +38.7 %; 85 % pp +30.3 %,
tg +39.6 %. Average request latency: 60 % 147.4 s → 116.6 s; 85 % 187.5 s → 141.3 s.

`trim-stage-weights` moves experts from RAM to VRAM: GPU caches 31.3 GiB (vs 27.8), RAM copy
22.4 GiB (vs 25.9), lend 3591/3591, file tier still 0.0. Layout: layers 0-25 on CUDA0 (x16),
26-47 on CUDA1 (x4), 8452 + 7972 slots. `pipeline-windows=2` keeps 272-284 MiB of CUDA0 out of the
expert cache; that is included in the numbers.

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

# 20261022 PARALLEL quick test (can the lane serve 2-4 requests at once?)

Question: agentic tooling (OpenCode subagents) can issue several requests at once; today llama-swap
queues them on the engine's single slot. Does Strata's `PARALLEL=N` (engine `--batch N`, batch slots)
pay for itself at 262144 ctx / F16 KV? Priority: speed ("serve more requests but keep decent speed");
256K per request is a hard requirement.

## Engine facts (measured from the start log; they replace the plan's estimates)

+ `PARALLEL=2` → engine `--batch 2` and **`--pipeline-windows 2 is off: not with --batch slots`** —
  the tuning's biggest decode win is structurally incompatible with any batch slot count, so
  single-stream is capped at ≈ 67.35 / 1.266 ≈ 53 tg before anything else.
+ Each slot carves **one extra full-262144 F16 session per stage**: `2 slot sessions on CUDA0
  (3.32 GiB each)`, same on CUDA1 — `--batch 2: the slot sessions take 6.64 GiB of VRAM on CUDA0
  that the expert cache would otherwise hold`. The slots are carved **before** the expert cache is
  sized: CUDA0 8450 → 4936 slots (15.31 → 8.95 GiB), CUDA1 7972 → 4736 (15.96 → 9.50 GiB), i.e.
  −41 % / −40 %; the RAM copy grew 22.37 → 28.00 GiB to compensate.
+ Batch windows decode **without MTP drafts** by default (the c=2 rows report draft 0 / accepted 0):
  each concurrent slot also loses the speculative-decoding multiplier.

## Numbers (60 % of 262144 ctx, out 2621, 1 rep, direct containers)

+ parallel=1 (control)
  + pp/tg/TTFT/latency: 2103.9 / 63.9 / 74.7 / 115.6

+ parallel=2 but only 1 req
  + pp/tg/TTFT/latency: 1766.7 / 39.6 / 89.0 / 154.9

+ parallel=2 and 2 req
  + req1 pp/tg/TTFT: 1732.7 / 29.2 / 90.7
  + req2 pp/tg/TTFT: 1706.6 / 42.4 / 91.6
  + Total time: 253.7 s

A lone request on the PARALLEL=2 lane loses **38 %** tg, 16 % pp and +19 % TTFT, paid 24/7 whether
or not anyone uses the second slot. The loss is pipeline-off plus the −41 % expert cache; without
drafts the pipeline ceiling alone would have been ~53 tg.

Concurrency does not win on its own terms either: two simultaneous requests on the PARALLEL=2 lane
produced 20.7 out-tok/s aggregate over their wall time, below the PARALLEL=1 lane serving one
request end-to-end (22.7 t/s); two sequential PARALLEL=1 reps (~231 s) would finish both requests
faster than serving them concurrently on PARALLEL=2 (253.7 s). Per-request tg 29.2 / 42.4 vs 63.9
queued.

Verdict: Keep `PARALLEL=1` on both lanes. Queue subagents on the single slot (llama-swap's
`concurrencyLimit: 20` already does); prefix caching plus the 4 conversation-parking slots make
queueing cheap for agent turns (turn increments reuse parked prefixes). The accepted cost is
head-of-line blocking: a 100-200K-token compaction stalls the queue for its ~1-4 min prefill, but
still better than halving everyone's speed.

Revisit only if Strata ships `--batch` + `--pipeline-windows` compatibility, or a KV-streaming mode
that keeps the expert cache intact while slots exist (that would fix the eviction but not the 53 tg
pipeline ceiling), or the workload truly needs 2+ simultaneous long decodes.

# 20261023 Upgrade to Strata 0.2.3 (engine 0.1.48)

Setup/deploy mechanics are in [`notes/strata.md`](../strata.md). First start dropped the 0.2.2
install record, ran setup once, and reused the pack and MTP (SC117 swapped in with its own mmproj).
Spot-check (`run_longctx.py --ctx 262144 --out-pct 1 --in-pcts 60,85`, 2 reps, through llama-swap,
vanilla lane):

+ 60 %: pp 2077.3 / tg 65.95 (accept 0.654)
+ 85 %: pp 2132.5 / tg 63.30 (accept 0.670)

Within ±2 % pp and −2 %/−5 % tg of the 0.2.2 tuning — **no RX 7000 regression from the 0.1.48
rebuild** (upstream claims none; the Strix Halo changes are gfx1151-only), so treat the tg delta as
run-to-run noise unless it repeats.

# 20261024 Upgrade to Strata 0.2.4.2

Switched to our own `0.2.4.2-rocm-gfx110x` image (engine 0.1.48 unchanged; install record kept,
straight to serving with no setup pass). Spot-check (`run_longctx.py --ctx 262144 --out-pct 1
--in-pcts 60,85`, 2 reps, through llama-swap, vanilla lane):

+ 60 %: pp 2100.5 / tg 64.05 (accept 0.654)
+ 85 %: pp 2172.4 / tg 64.00 (accept 0.673)

+1.1 %/+1.9 % pp and −2.9 %/+1.1 % tg against 0.2.3 — both noise, no action.

# 20261025 Upgrade to Strata 0.2.5 (engine 0.1.50 and re-tuned hipBLASLt table)

0.2.5 moved the engine to 0.1.50 (install record dropped, setup once with no repack/download) and
re-tuned the packaged hipBLASLt table (`100200`). Two `run_longctx.py` samples (`--ctx 262144
--out-pct 1 --in-pcts 60,85`, 2 reps, through llama-swap, vanilla lane):

| Sample | 60 % pp / tg (accept) | 85 % pp / tg (accept) |
| --- | --- | --- |
| Sample 1 | 2005.8 / 63.55 (0.649) | 2049.4 / 65.75 (0.692) |
| Sample 2 | 1936.1 / 63.65 (0.656) | 2074.8 / 64.30 (0.683) |

The 0.2.5 60 % average is dragged down by the first measured rep of each sample (1900.8 and
1739.0 t/s); those requests logged an ~8.3-8.7 GB OS read from NVMe (cold page cache after the
container churn). The steady reps are 2110.8 / 2133.2 (60 %) and 2041.2-2075.0 (85 %), inside the
0.2.4.2 range. `cache_n` 0 on every rep.

A direct fixed-prompt A/B (600,000 chars ≈ 147K tokens, `max_tokens=1`, 3 reps, same container
recipe) separates engine/table from run variance:

| Arm | pp at ~147K | pp at ~222K |
| --- | --- | --- |
| 0.2.5 + new table (default) | 2110.5 t/s | 2116.1 / 2152.8 t/s |
| 0.2.5 + 0.1.48 table (`STRATA_HIPBLASLT_TUNING`) | 2106.1 t/s | — |
| 0.1.48 (20261024 run_longctx, for scale) | 2100.5 t/s | 2172.4 t/s |

Verdict: the 0.1.50 engine and the re-tuned table are neutral for these lanes — the direct numbers
match 0.2.4.2 and the old table within 0.2 %; the run_longctx 85 % gap above is cold-page and run
variance, not a regression. tg and acceptance are within noise. No action.

Smoke through llama-swap on 2026-10-25: keyless chat, `get_weather{"city":"Paris"}`, `max_tokens:-1`,
640×360 vision ("Red") and the `:npt` alias answered on vanilla; SC117's setup picked its own mmproj
(`2e788f8c…`), answered a chat, a tool call and a 640×360 picture ("Blue"). Engine landmarks match
0.2.4.2 on both lanes: 22.37 GiB resident, lend 3591, 31.3 GiB GPU caches, 0.0 GiB file tier, MTP
1348 MiB, `prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)`. The engine log
adds `this CPU has no AVX-512: the expert kernels run on AVX-2` (the 0.1.50 CPU-row work does not
reach this host).

# 20261027 Vision budget 1024-8192 (declarative)

Both lanes now read pictures at up to 8192 image tokens, with small ones scaled up to 1024
(`--max-tokens 8192 --min-tokens 1024`; setup's default was 1024 max, no floor) — the same range as
the Qwen3.8 27B llama.cpp lanes. The budget is set per lane in llama-swap
(`VISION_TOKENS`/`VISION_MIN_TOKENS`) and written into the run settings by entrypoint fork change 5,
so no hand-edited config is involved. Mechanism and token math: [`notes/strata.md`](../strata.md).

Verified on SC117 (2026-10-27, llama-swap lane):

- Entrypoint log `Vision budget: strata-iq3_s.json: max_tokens=8192, min_tokens=1024.`; the record
  carries both keys; the reader runs `strata-vision-hip ... --gpu --max-tokens 8192 --min-tokens 1024`.
- Reader warm-up (2048×2048) went from 1024 tokens / 0.4 s to **4096 tokens / 3.5 s** — mtmd got the
  new cap (the warm-up picture is below 8192, so it is not scaled down).
- A 3840×2160 (8.29 MP) picture became **8160 image tokens** (120×68 at 32 px per side, 1024 px per
  token) and the request reported `prompt_tokens 8223` (8160 image + 63 text), no refusal. First
  encode + prefill: 27.3 s wall (engine prefill 8.1 s at 1016 t/s), same picture again: 1.3 s wall,
  `cached_tokens 8218`.
- VRAM: the engine shifted residents 22.54 → 23.17 GiB to RAM for the bigger reader warm-up (28 GiB
  budget unchanged). One soft `only 450 MiB of VRAM free at the start of this request (below 512 MiB)`
  warning appeared on a ~21k-token text request before any 8K image; decode/json stayed at 70-100 t/s
  and prefill ~1000 t/s. Cards run near-full by design (expert cache). If it recurs, raise
  `vram-reserve-mib` 1536 → 1600-2048 in `STRATA_SET` (engine's own suggestion) and re-check.

# 20261028 0.2.7 engine 0.1.52 (staged HC read; KVA rejected; prefill-pipe candidate)

The image moves to Terminal 0.2.7 / engine 0.1.52 — the first Qwen engine byte change since 0.1.50
(engine folder `0.1.52-linux-x86_64-hip-rx7900`, so the lane re-runs setup once and reuses its pack;
deployment record: [notes/strata.md](../strata.md)).

Engine changes that reach gfx1100: only the staged/split hyper-connection read (`STRATA_HC_SPLIT`,
unset = staged). It is picked by a per-card self-test at every start and used only when exact;
`STRATA_HC_SPLIT=0` returns the plain read. Watch the start log for the chosen variant line. The rest
of the 0.2.7 release is CUDA- or gfx12-specific (12 GB route prior, sm_80+ int8 prompt attention,
layer-major/pipelined CUDA split, the four gfx12 decode switches); the PLE residency auto-rule does
not admit the 28.8 GB table on this 62 GiB host, so `--ple-io ram` stays out of reach.

Candidates and rejects (parameter rationale for this lane):

- **`STRATA_KVA=1` ("Faster long prompts"): rejected for now.** It reads the last 12 layers of a
  prompt of 4,096+ tokens approximately (+11 % at 14K, +18 % at 30K on an RTX 5070); upstream keeps
  it off by default because earlier settings dropped records from long lists and failed agent tasks,
  and even the shipped setting (layers 36-47 approximate, last 2,048 tokens exact) found 585 of 606
  facts. Coding and agentic loops are this lane's job: do not enable it without a fact-retrieval and
  agent-loop A/B that passes cleanly.
- **`--prefill-pipe` on the two-card layer split: candidate, not yet run.** Upstream measured
  +15 % at 4K and +23 % at 8K prompts on two RTX A4000; unmeasured on HIP. A/B it after the 0.1.52
  baseline is recorded (same prompts, rotate the arms).

## Measured curve (0.1.52, 2026-10-28)

`run_longctx.py --ctx 262144 --out-pct 1 --reps 2`, through llama-swap, vanilla lane; `cache_n` 0 on
every rep. Startup landmarks: `prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version
100200)`, staged HC read on CUDA0 and CUDA1 (bit-exact self-test), 22.62 GiB resident RAM copy,
lend 3591, expert caches 8452 + 7849 slots.

| in % | pp t/s | tg t/s | accept |
| ---: | ---: | ---: | ---: |
| 20 | 1564.5 | 66.05 | 0.641 |
| 40 | 1882.9 | 66.65 | 0.651 |
| 60 | 2063.9 | 64.30 | 0.653 |
| 85 | 2121.1 | 61.60 | 0.665 |

Against the 0.2.5/0.1.50 60/85 table above: pp within +1 to +3 % (inside the 1936-2133 t/s spread),
tg at 85 % 61.6 against 64.3-65.8 (−4 %, accept 0.665 against 0.68-0.69) — small enough for two reps
to be noise, but it repeats the direction at 60 %, so re-check with the next paired run before
calling it an engine delta. The 0.1.52 package ships the **0.1.48 hipBLASLt table bytes** again
(`1bb8cd2e…`, 845 B), not the 0.1.50 re-tune — the pp numbers above are therefore a fresh baseline,
not an A/B against the 2026-10-25 table spot-check. Targets hold with room: pp ≥ 1000 t/s and
tg ≥ 25 t/s at 85 % of 262144.

No 0.1.52 regression on the picture path: a 420×420 red/blue-halves picture answered correctly
through the swap (reasoning "pure red and pure blue", 1110 prompt tokens at the 8192 budget), and the
reader warm-up still logs `warmed up at 4096 image tokens`.

