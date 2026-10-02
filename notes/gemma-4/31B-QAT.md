# 20260804

Testing Gemma 4 31B QAT with MTP decoding: `--spec-type draft-mtp --spec-draft-n-max 2`

With everything loaded onto the GPUs, both cards use 23.902/23.984 GB of VRAM. There is not much headroom, considering I have a monitor plugged in.

@211K context:
No MTP:  pp 368.71 t/s, tg 15.8 t/s (Not limited by VRAM capacity)
Has MTP: pp 254.14 t/s, tg 6.65 t/s (Limited by VRAM capacity; llama.cpp competes with GNOME for VRAM, causing swapping)

Gemma 4 MTP cannot be offloaded to the CPU because it shares the main model's KV cache.
So the only way to free some VRAM is to offload layers to the CPU.

Same 211K test:

Everything on GPU: pp 254.14 t/s, tg 6.65 t/s
62 layers on GPU:  pp 254.74 t/s, tg 6.67 t/s

Conclusion: Not enough VRAM for MTP.

# 20260922

Keep mmproj (vision) on CPU, free some VRAM for bigger ubatch size for faster pp.

Original (ub 1280):
  + 20%: pp  732.34 t/s  tg 21.59 t/s
  + 40%: pp  513.92 t/s  tg 19.24 t/s
  + 60%: pp  389.69 t/s  tg 17.32 t/s
  + 85%: pp  301.06 t/s  tg 15.56 t/s

The only downside: requests with images will be slow since image tokens are calculated by CPU.
For example, at 20% ctx, we get pp at 800+ t/s, but sending a request with 1 image at 2k ctx (1%)
drops pp to about 100 t/s. Acceptable I think.

New result (ub 1664):
  + 20%: pp  842.46 t/s  tg 21.14 t/s
  + 40%: pp  606.51 t/s  tg 18.95 t/s
  + 60%: pp  465.70 t/s  tg 17.13 t/s
  + 85%: pp  362.76 t/s  tg 15.40 t/s

# 20261001 Host-memory and checkpoint measurements

Keep F16 K/V caches, 256K context, one slot, CPU vision projector, ubatch 1664,
GPU split and speculative decoding disabled. KV-cache quantization is excluded
because coding/agentic reliability is more important than saving VRAM.

On `b11096-c550d2f60`, a synthetic 224386-token Python-source fixture with 22
appended review turns measured cold prefill at 359.42 t/s and median one-token
appended-turn latency at 1.451 s. Editing recent history reused 224912 tokens,
processed 64 and took 1.757 s. The separate 128-token decode was 15.32 t/s.
This is a serving/memory test, not a model-quality benchmark.

Partial context checkpoints were **800.013 MiB each** on this build. They save
SWA state, not all global-attention K/V. The existing 20-checkpoint configuration
peaked at 37.31 GiB of cgroup memory: maximum anonymous memory was 19.99 GiB,
while about 17.26 GiB was reclaimable file cache. No swap or OOM events occurred
in the unconstrained baseline. Test the 32 GiB Docker limit separately rather
than treating charged file cache as irreducible checkpoint memory.

With `b11312-0c1e57098`, 16 checkpoints, RAM prompt-cache disabled, and Docker
`--memory 32g --memory-swap 32g`, the same 224386-token/22-turn workload completed
without swap or OOM events. Maximum anonymous host memory was 16.97 GiB, about
3.02 GiB below the 20-checkpoint baseline. The cgroup limit reclaimed file cache.
Cold prefill was 360.94 t/s, median appended-turn latency 1.450 s, recent-edit
latency 1.790 s, and 128-token decode 15.29 t/s: no material speed change.
The edit still reused 224912 tokens and processed only 64.

Use a separate 31B checkpoint macro with count 16 and the existing minimum
spacing 13107. This leaves more anonymous-memory headroom; it does not shrink
the 256K context, but editing older history can require more re-prefill.

Plain/streamed tool-result loops, strict JSON-schema output and a synthetic
red-image check passed. These are short compatibility checks, not evidence of
unchanged coding quality over long agentic runs. The tested ROCm image is pinned
and normal logging uses `-lv 3`; all inference-precision and offload settings stay
unchanged. CPU image processing remains slow: the small image check took 16.83 s.

The final deployed config was rechecked through llama-swap: all four smoke
checks passed with the 16-checkpoint cap, F16 defaults and 32 GiB/no-swap limits.
No container swap or OOM events occurred.

# 20261002 Ubatch and vision-placement tuning

Ubatch `1664 → 768` and CPU → GPU vision; retain split `31,30`,
batch 2048, one private 256Ki slot and F16 KV.

| Ubatch / projector | Cold prompt tokens | pp (t/s) | tg (t/s) | Peak VRAM (GiB) |
| --- | ---: | ---: | ---: | ---: |
| 1664 / CPU baseline | 227692 | 333.79 | 15.32 | 23.755 / 23.739 |
| 768 / CPU | 227692 | 350.29 | 15.52 | 21.249 / 21.264 |
| 768 / GPU, ~85% | 222676 | 357.80 | 15.76 | 23.022 / 21.397 |
| 768 / GPU, near limit | 254818 | 323.35 | 14.87 | 22.896 / 21.271 |

Smaller ubatch freed ~2.5 GiB/card with similar speed and ~1.5 s cached latency;
peak anonymous RAM fell 17.238 → 15.373 GiB. GPU vision spent much of that saving:
minimum measured headroom was **0.962 / 2.587 GiB**.

Corrected near-limit validation passed at 97.21% context with cached turns,
recent edits, decode and vision. Three console screenshots passed seven-field
extraction with ~223000 history plus ~11000 description tokens; cached portrait/1080p
latency fell **25.65 / 23.78 s → 11.57 / 11.21 s** on GPU, including generation.
Ubatch must fit the 560-token image chunk, not the surrounding description.

Tool/analysis checks passed, but follow-up and restoration fully re-prefilled
long histories. The novel-summary check did not score grounded correctness.
Post-sync plain/streamed tool loops, JSON schema and vision passed (short image:
2.84 s).

Single synthetic runs on digest-pinned `b11312-0c1e57098`, not general quality
benchmarks. VRAM: 100-ms peaks, GPU 0 (CPU x16) / GPU 1 (chipset x4). Container
swap/OOM stayed zero; host swap increased and heavy desktop load remains untested.
