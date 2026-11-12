+ Re-test DFlash2 on the Flash-Next lanes when a later Strata engine can run it beside
  `--pipeline-windows` (or lifts its layer-split decode): on 0.2.10/0.1.56 DFlash2 alone was 16-18%
  slower decode here (notes/qwen3.8/Flash-Next.md 20261103), and the two cannot be combined.
+ Bring up the static uncensored DeepSeek lane (Abiray Uncensored-GSQ-RCO-IQ2_XXS) when disk allows - the
  replacement for the removed direction ablation on uncensored traffic that needs thinking
  (notes/deepseek/v41f1017.md 20261112; method and the reported numbers: notes/uncensored.md):
+ Optional, GLM only: the shipped GLM direction's think flag is not in its metadata (`fitted_on: Maya-S24`
  only), and upstream's GLM thinking data is on the XS-fitted direction (0/24 refusals at 1.5, 4/24 token-capped).
  Run `ablate_fit.py eval --think on` against `glm-5.3-flash-iq2s` if GLM uncensored-under-thinking ever matters.
