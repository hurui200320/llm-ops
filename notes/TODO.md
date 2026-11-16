+ Re-test DFlash2 on the Flash-Next lanes when a later Strata engine can run it beside
  `--pipeline-windows` (or lifts its layer-split decode): on 0.2.10/0.1.56 DFlash2 alone was 16-18%
  slower decode here (notes/qwen3.8/Flash-Next.md 20261103), and the two cannot be combined.
+ Optional gfx1100 experiment on 0.2.14 (engine 0.1.60): `STRATA_PF_P1A=1` — the P1A prompt experts
  (F16 matrix cores, F16 activations/H) are default-on only on Windows gfx1151, but the device code
  compiles for gfx1100 and the switch is read on HIP; upstream never measured it on gfx1100. A/B pp
  at 60/85% and check quality before keeping.
+ Optional A/B: `"draft_costs": true` in a lane's run config (the repeatable-output draft-cost
  table). Default on Metal only since 0.2.14; upstream measured CUDA code/chat −1.5% / JSON +1.9%
  against live costs, HIP untested. Only worth it if tg drifts after the fused-kernel switch.
+ Optional, GLM only: the shipped GLM direction's think flag is not in its metadata (`fitted_on: Maya-S24`
  only), and upstream's GLM thinking data is on the XS-fitted direction (0/24 refusals at 1.5, 4/24 token-capped).
  Run `ablate_fit.py eval --think on` against `glm-5.3-flash-iq2s` if GLM uncensored-under-thinking ever matters.
+ Bench abiray deepseek?
