+ Try `--spec-draft-n-max 3` for Qwen 3.8 27B
+ Test muse glimmer extend to 256K, community and unsloth reports 256K fine, but meta suggest 128k
  + Or maybe considering remove it?
+ Re-test DFlash2 on the Flash-Next lanes when a later Strata engine can run it beside
  `--pipeline-windows` (or lifts its layer-split decode): on 0.2.10/0.1.56 DFlash2 alone was 16-18%
  slower decode here (notes/qwen3.8/Flash-Next.md 20261103), and the two cannot be combined.
+ Re-bench longctx on deepseek as it now moves to project maya.
+ upgrade maya to 1.1.55
+ bench the GLM-5.3-Flash Maya-M-Derisked lane
  + if int8 kv not good, revert with `--kv fp16` and clear `/data/pack/slots` (the slot meta does not record the KV format)