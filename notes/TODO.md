+ Try `--spec-draft-n-max 3` for Qwen 3.8 27B
+ Test muse glimmer extend to 256K, community and unsloth reports 256K fine, but meta suggest 128k
  + Or maybe considering remove it?
+ Re-test DFlash2 on the Flash-Next lanes when a later Strata engine can run it beside
  `--pipeline-windows` (or lifts its layer-split decode): on 0.2.10/0.1.56 DFlash2 alone was 16-18%
  slower decode here (60% pp/tg 2100.5/54.80, 85% 2170.9/52.65 vs MTP 1993.1/65.00, 2119.2/64.30;
  notes/qwen3.8/Flash-Next.md 20261103), and the two cannot be combined. Current arm: MTP +
  pipeline-windows=2 + DRAFTING=mtp; the DFlash2 packs stay on both data trees for the retest.
+ Re-bench longctx on deepseek as it now moves to project maya.
+ upgrade maya to 1.1.54
