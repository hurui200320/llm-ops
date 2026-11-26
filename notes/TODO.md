+ Re-test DFlash2 on the Flash-Next lanes when a later Strata engine can run it beside
  `--pipeline-windows` (or lifts its layer-split decode): on 0.2.10/0.1.56 DFlash2 alone was 16-18%
  slower decode here (notes/qwen3.8/Flash-Next.md 20261103), and the two cannot be combined.
+ Optional, GLM only: the shipped GLM direction's think flag is not in its metadata (`fitted_on: Maya-S24`
  only), and upstream's GLM thinking data is on the XS-fitted direction (0/24 refusals at 1.5, 4/24 token-capped).
  Run `ablate_fit.py eval --think on` against `glm-5.3-flash-iq2s` if GLM uncensored-under-thinking ever matters.
+ Bench abiray deepseek? (coding test)
+ Inkling 2 small? AA score 31, while deepseek v41f 1017 score 32, model cap ranked at 97, where the only model ranked after it is qwen3.8 fn, which is 106. Vision might be the best, better than Gemma 4 and Qwen3.8 Flash Next, good for reading comics and focus on details. Need test.
+ Upgrade llamacpp, strata and project maya
+ Qwen3.8 Flash Next 512K: deployed 2026-11-27 on both lanes — `CONTEXT=512k` (setup
  derives `--rope-scaling yarn --rope-scale 2` itself; do not pin rope flags in `STRATA_SET`) +
  `kv=int8` (setup's install default, replacing the `kv=fp16` override). Record and mechanics:
  notes/qwen3.8/Flash-Next.md 20261127. **Open: re-run the full bench suite (512k speed, tool
  call, reasoning, coding) and keep the change only if there is no regression** — INT8 KV is
  provisional under the AGENTS.md KV-quant rule (official API uses FP8, not INT8).
