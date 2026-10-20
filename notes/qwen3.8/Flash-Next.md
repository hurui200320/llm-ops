# 20261019 First try

Normally I won't run Qwen 3.8 Flash Next on my setup because either I have to go with lower quant
and it will be slow since I have to offloading stuff to CPU. Last time running Laguna S 2.1 doesn't
give a good baseline.

However, I was suggested with https://github.com/niko1221/strata. This software does only one thing:
run Qwen3.8 Flash Next. I decided to go with IQ3_S + 256K context + CPU vision.

Strata is different from llama cpp, where llama server just split layers to different devices, strata
can allocate stuff on per-expert level, so it can performs better than llama server.

However, strata can't just load the GGUF and run, it requires some special pre-process.
The models are downloaded to qwen3.8 folder, and then copied to strata folder.
This is the folder required by strata, we build once and save to NAS.

Also we need to build our own docker image as Strata doesn't provide them.

# 20261020 Plain llama.cpp baseline (before Strata)

Quick test on fedora-tuf (desktop running, 2026-10-20): can vanilla llama.cpp run Flash-Next at
all, and what does it cost? Measured so Strata's numbers have a comparison point. Both lanes +
vision, single 256K slot, 1 rep per level (cold prefill, `cache_n=0`, 1% output cap = 2621
tokens), through a bare `llama-server` (llama-swap untouched).

## Setup

+ Image `ghcr.io/ggml-org/llama.cpp:full-rocm`, `version: 0.5.0-dev` (build 11374, commit
  `b92761a51`, Oct 3) — knows the `qwen4exp` arch (48 blocks, 512 experts, hybrid SSM/attention)
  and the `qwen3vl_merger` vision projector.
+ 1 slot, `--ctx-size 262144`, f16 KV, `--flash-attn on`, `-b 2048 -ub 512`, `--cache-ram 0`,
  default threads (8), sampling from the GGUF (`--temp 1.0 --top-p 0.95 --top-k 20`),
  `--jinja --reasoning on --reasoning-preserve`; vision loaded **on GPU** (llama.cpp default
  offload), `--image-min-tokens 1024` — verified with a real image request on both lanes
  (~280-330 t/s image prefill, 10-14 s wall for one small image + 120-token answer).
+ Weights (≈78-80 GiB) can't fit 2×24 GB: `-ngl 99` + expert layers pushed to CPU with one
  **comma-separated** `-ot` (passing `-ot` twice keeps only the last value — learned the hard
  way):
  `blk\.([0-7]|2[4-9]|3[0-2])\.ffn_(up|down|gate)_exps\.weight=CPU,per_layer_token_embd=CPU`
  → both lanes keep 31 of 48 expert layers on GPU (17 on CPU; IQ3_S experts 0-7 + 24-32 on CPU,
  ORCA 0-8 + 24-31); `--tensor-split 50,50`; the 26.8 GiB `per_layer_token_embd` (PLE) sits on
  CPU via the lazy-read/mmap path either way. CPU-resident weights: IQ3_S ≈ 16.6 GiB experts +
  26.8 GiB PLE; ORCA ≈ 17.6 + 26.8 GiB.
+ **VRAM margin lesson**: "leave ~100 MB free per card after startup" does not survive the first
  request — llama-server allocates more when a request hits (per-stream hipBLAS handles, vision
  encoder buffers). At ~70 MB free, the first image request aborted with
  `CUBLAS_STATUS_ALLOC_FAILED` (`clip_encode`). Working margins are **0.2-1.1 GiB free per card
  after a text+vision warmup**. Also: the llama-swap macro's `--memory 32g` is too small for this
  model — the CPU-resident weights need host RAM (servers ran with `52g`).

## Results

Both lanes: 256K ctx, vision loaded, 1 rep per level.
Values are IQ3_S (vanilla, ISTA-DASLab) / ORCA uncensored IQ3_XXS (same config, 31 GPU expert layers):

+ 20%: pp 424.54 / 412.70 t/s, tg 16.55 / 16.65 t/s
+ 60%: pp 338.62 / 331.43 t/s, tg 14.34 / 14.39 t/s
+ 85%: pp 298.58 / 294.71 t/s, tg 13.29 / 13.34 t/s

MTP A/B (uncensored lane only — the vanilla GGUF ships no MTP head), pp / tg t/s at 20 / 60 / 85%:

+ no MTP (above):
    pp 412.7 / 331.4 / 294.7, tg 16.65 / 14.39 / 13.34
+ MTP on GPU (needs 4 expert layers evicted: 27 GPU experts):
    pp 295.3 / 249.6 / 225.9, tg 16.17 / 13.34 / 11.74, accept 0.44-0.50
+ MTP on CPU (30 GPU expert layers — one more CPU layer than the GPU-draft run):
    pp 234.2                , tg 6.46                 , accept 0.46 (20% only)

Verdict: MTP is a **no** on this rig. The draft (4.1 GB nextn GGUF: 1 MoE layer + its own
token_embd/output) doesn't fit VRAM (first attempt died allocating its 512 MiB KV); on CPU the
draft makes decode 2.5× slower; on GPU it costs 4 target expert layers and still comes out
slower than no-MTP on every level (accept rate only ~0.44-0.50).

CPU threads (IQ3_S lane, same placement; default is 8 — the 13700K's P-cores), pp / tg t/s at
20 / 60 / 85%:

+ 8 (default): 424.5 / 338.6 / 298.6, 16.55 / 14.34 / 13.29
+ 12:          413.8 / 335.9 / 297.8, 16.50 / 14.21 / 13.21
+ 14:          350.0                , 16.56                 (20% only, skipped)

Verdict: keep the default 8 — more threads don't help this placement (12 is ~1-2.5% slower on
prefill, 14 is ~18% slower; spilling onto E-cores / the box's power budget costs more than the
added throughput); decode is unaffected.

VRAM / RAM usage (peaks over the full session, sysfs + meminfo + cgroup; cards as PCH / display):

+ IQ3_S: free 930 / 880 MiB after load, 629 / 474 MiB after text+vision; peak used 23936 /
  24098 MiB; cgroup peak 19.9 GiB; host RAM avail min 52.3 GiB.
+ ORCA IQ3_XXS: free 623 / 665 MiB, 338 / 207 MiB after text+vision; peak used 24229 / 24365
  MiB; cgroup peak 27.8 GiB; host RAM avail min 52.7 GiB.
+ ORCA + MTP (GPU draft): free 1466 / 1608 MiB, 1102 / 1139 MiB after text+vision; peak used
  23494 / 23434 MiB; cgroup peak 19.4 GiB; host RAM avail min 52.0 GiB.
+ KV (f16, 262144): 6144 MiB total (12 full-attn layers × 512 MiB) + ~1.5 GiB recurrent states
  (0.75 GiB per card); compute buffers ~1.3 / 1.1 GiB per device; RAM never got tight — the
  CPU-side weights live in page cache (host cached peaked ~54 GiB, available never below
  51.5 GiB).

## Takeaways for the Strata comparison

+ Vanilla llama.cpp **does run Flash-Next** (both lanes, vision included) on the XTX pair, but
  pp 300-425 / tg 13-17 t/s is far off our own targets (pp ≥ 1000, tg ≥ 25 at 85%) and off
  Strata's upstream XTX numbers (≈1.5K pp, 59-65 tg) — exactly the gap Strata's VRAM+RAM expert
  residency is supposed to close. This is the number to beat.
+ Model-size effect is small with this placement: ORCA (bigger experts per layer) is ~2-4%
  slower than IQ3_S at 20-60% and ~1% at 85%.
+ Raw artifacts: `bench/results/longctx/` (gitignored): `iq3s-longctx.{json,log}`,
  `orca-longctx.{json,log}`, `orca-mtp-gpu-longctx.{json,log}`,
  `orca-mtp-cpu-draft.{json,log}`, the CPU-thread sweep
  (`iq3s-t12-longctx.{json,log}`, `iq3s-t14-partial.{json,log}`), plus the `*.tsv`
  VRAM/RAM samplers.
+ Not tested here: other quants/splits (±1-2 expert layers is the tuning granularity,
  thread counts 8-14 are settled), `--load-mode dio` (would bypass the page cache the CPU
  weights rely on).
