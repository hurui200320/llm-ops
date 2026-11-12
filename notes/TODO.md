+ Re-test DFlash2 on the Flash-Next lanes when a later Strata engine can run it beside
  `--pipeline-windows` (or lifts its layer-split decode): on 0.2.10/0.1.56 DFlash2 alone was 16-18%
  slower decode here (notes/qwen3.8/Flash-Next.md 20261103), and the two cannot be combined.
+ Investigate maya's ablation (one direction?) and normal abliterated/uncensored model
  + Also note when to use what, for example, heretic model may give better KLD with lower refusal, so worth redownload
  + Normal abliteration might just not worth for big model, in this case, maybe use the direction ablation in maya.
  + Also investigate reasoning vs non-reasoning? The maya's eval uses non-reasoning, but our usage is mainly reasoning.
  + note this in notes/uncensored.md
- Maya 1.1.59 shipped per-request refusal ablation; the DeepSeek lane's direction transfer is verified (99 % -> 3.8 %
  refusals, no coherence damage; notes/maya.md 20261112) and the vanilla GLM lane has the Maya-S24 direction wired.
  Close the investigation items above against it:
  + the reasoning-vs-non-reasoning question: the direction was fitted with think off (`maya.ablate.think: off`);
    run `ablate_fit.py eval --think on` against the DS lane to check refusals/quality under thinking.
  + a GLM direction fitted on GLM-5.3-Flash exists; heretic-style per-model fits remain an option if a shipped
    direction underperforms (`ablate_fit.py fit` works on GLM-5.3-Flash now).
  + GLM lane ablated-request tg cost, quantified 20261112: pp free, tg -4.0 % (40 %) / -4.8 % (85 %) on requests
    that ask for ablation only - plain requests are bit-identical (notes/maya.md 20261112). Acceptable as an
    opt-in; revisit only if a fitted GLM-specific direction changes the projection cost.
  + `MAYA_GLM_ROUTE_AFFINITY` (1.1.59's GLM Turbo knob, lossy: prefers VRAM-resident experts, changes answers)
    as a screening arm on the GLM lanes - tg is the weak axis there (notes/glm/5.3-flash.md).
