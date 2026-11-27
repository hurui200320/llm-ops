+ Re-test DFlash2 on the Flash-Next lanes when a later Strata engine can run it beside
  `--pipeline-windows` (or lifts its layer-split decode): on 0.2.10/0.1.56 DFlash2 alone was 16-18%
  slower decode here (notes/qwen3.8/Flash-Next.md 20261103), and the two cannot be combined.
+ Inkling 2 small? AA score 31, while deepseek v41f 1017 score 32, model cap ranked at 97, where the only model ranked after it is qwen3.8 fn, which is 106. Vision might be the best, better than Gemma 4 and Qwen3.8 Flash Next, good for reading comics and focus on details. Need test.
+ Upgrade llamacpp and project maya
+ Bench abiray deepseek? (coding test)
+ Bench GLM with ablation alias? Also remove the derisked lane bench result
  + speed, tool call, reasoning, coding
+ Qwen3.8 Flash Next 512K: bench coding and keep the change only if there is no regression