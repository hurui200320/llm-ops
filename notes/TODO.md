+ Try `--spec-draft-n-max 3` for Qwen 3.8 27B
+ Test muse glimmer extend to 256K, community and unsloth reports 256K fine, but meta suggest 128k
  + Or maybe considering remove it?
+ DFlash2 vs MTP longctx A/B on the Flash-Next lanes (0.2.10): keep DFlash2 unless the 262144 spot
  (60/85%) regresses against the 0.2.8 MTP baseline (60% pp 2038.8 / tg 65.50, 85% pp 2122.1 /
  tg 62.30); fallback is `pipeline-windows=2 spec-min-p=0.7` + `DRAFTING=mtp`
  (deploy/llama-swap.config.yaml). Numbers go to notes/qwen3.8/Flash-Next.md.
