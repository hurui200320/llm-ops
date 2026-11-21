+ Re-test DFlash2 on the Flash-Next lanes when a later Strata engine can run it beside
  `--pipeline-windows` (or lifts its layer-split decode): on 0.2.10/0.1.56 DFlash2 alone was 16-18%
  slower decode here (notes/qwen3.8/Flash-Next.md 20261103), and the two cannot be combined.
+ Optional, GLM only: the shipped GLM direction's think flag is not in its metadata (`fitted_on: Maya-S24`
  only), and upstream's GLM thinking data is on the XS-fitted direction (0/24 refusals at 1.5, 4/24 token-capped).
  Run `ablate_fit.py eval --think on` against `glm-5.3-flash-iq2s` if GLM uncensored-under-thinking ever matters.
+ Bench abiray deepseek?
+ Build `maya-hy4` (Tencent Hy4-Flash) into the Maya image once the model has an external score worth
  the eval (AA score pending): the engine exists upstream since 1.1.68 and Maya-XS fits this host
  (~110 GB RAM+VRAM), but it is one-request-at-a-time and upstream measures it slower than maya-ds4
  on AMD for now (upstream CHANGELOG v1.1.68-1.1.70; not built in the 1.1.70 bump).
