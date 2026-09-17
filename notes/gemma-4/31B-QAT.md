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
