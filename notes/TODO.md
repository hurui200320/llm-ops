+ Re-test DFlash2 on the Flash-Next lanes when a later Strata engine can run it beside
  `--pipeline-windows` (or lifts its layer-split decode): on 0.2.10/0.1.56 DFlash2 alone was 16-18%
  slower decode here (notes/qwen3.8/Flash-Next.md 20261103), and the two cannot be combined.
+ Re-bench longctx on deepseek as it now moves to project maya.
+ bench the GLM-5.3-Flash Maya-M-Derisked lane
  + if int8 kv not good, revert with `--kv fp16` and clear `/data/pack/slots` (the slot meta does not record the KV format)
+ bench the vanilla GLM-5.3-Flash Maya-M lane (`glm-5.3-flash-iq2s`): settings copied 1:1 from the
  derisked lane, functional gates passed 2026-11-06; no speed ladder yet (notes/maya.md 20261106)
+ Upgrade strata 0.2.12
  + after rollout: A/B `STRATA_ADAPT_FETCH=1` and `=2` on the vanilla lane (fetched PCIe misses go into the
    GPU cache; our ~11k CPU experts are read over the x4 PCH link)
+ Apply chat template from this repo for qwen3.8fn and glm53f, maybe also ds?
