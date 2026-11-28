+ Re-test DFlash2 on the Flash-Next lanes when a later Strata engine can run it beside
  `--pipeline-windows` (or lifts its layer-split decode): on 0.2.10/0.1.56 DFlash2 alone was 16-18%
  slower decode here (notes/qwen3.8/Flash-Next.md 20261103), and the two cannot be combined.
+ Inkling 2 small? AA score 31, while deepseek v41f 1017 score 32, model cap ranked at 97, where the only model ranked after it is qwen3.8 fn, which is 106. Vision might be the best, better than Gemma 4 and Qwen3.8 Flash Next, good for reading comics and focus on details. Need test.
+ Upgrade llamacpp
+ Project Maya 1.1.84 DeepSeek prefill regression (accepted 20261128, still above target): pp 40 % 1,368.4 ->
  1,207.0, pp 85 % 1,260.0 -> 1,170.1 (means; interleaved same-hour 1.1.74 control reproduced the old
  numbers), tg and answers unchanged. Top suspect #1252 (AVX-512 prompt-rows reuse 4 -> 16 tokens; this lane
  computes ~71 % of prompt experts on the CPU lane), then #1164 (RDNA warp sums) and #1143 (ring band). Options:
  build 1.1.84 with 79bd6658 (the #1252 commit) reverted and re-run `ab84_point.sh b84 r1` against a stock
  b84 point, or report upstream with the numbers. Record: notes/maya.md 20261128.
+ Bench GLM with ablation alias? Also remove the derisked lane bench result
  + speed, tool call, reasoning, coding
