+ Re-test DFlash2 on the Flash-Next lanes when a later Strata engine can run it beside
  `--pipeline-windows` (or lifts its layer-split decode): on 0.2.10/0.1.56 DFlash2 alone was 16-18%
  slower decode here (notes/qwen3.8/Flash-Next.md 20261103), and the two cannot be combined.
+ Investigate maya's ablation (one direction?) and normal abliterated/uncensored model
  + Also note when to use what, for example, heretic model may give better KLD with lower refusal, so worth redownload
  + Normal abliteration might just not worth for big model, in this case, maybe use the direction ablation in maya.
  + Also investigate reasoning vs non-reasoning? The maya's eval uses non-reasoning, but our usage is mainly reasoning.
  + note this in notes/uncensored.md
