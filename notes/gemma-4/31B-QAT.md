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

