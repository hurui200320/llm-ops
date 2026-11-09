+ Re-test DFlash2 on the Flash-Next lanes when a later Strata engine can run it beside
  `--pipeline-windows` (or lifts its layer-split decode): on 0.2.10/0.1.56 DFlash2 alone was 16-18%
  slower decode here (notes/qwen3.8/Flash-Next.md 20261103), and the two cannot be combined.
+ Upgrade maya to 1.1.57
+ Maya 1.1.57 GLM tuning arms (after the cutover): `MAYA_GLM_NGRAM_DRAFT=1` (drafts repeated
  history instead of the MTP block; Flash-eligible, default off) and `MAYA_GLM_LANE_ROW4=1`
  (4-row CPU-lane GEMV, bit-exact, x86 AVX2/FMA/F16C; matters only if the CPU lane is active).
  Low priority: `MAYA_GLM_VRAM_EVICT=lfu|lru` (new heat-evict policy, default off) - tg on these
  lanes is bounded by the RAM expert tier and SSD reads, not VRAM eviction. `MAYA_GLM_PROMPT_CACHE_DIR`
  is glm-dsa (744B) only, not available to the Flash lanes (notes/maya.md 20261110).
