+ Try `--spec-draft-n-max 3` for Qwen 3.8 27B
+ Test muse glimmer extend to 256K, community and unsloth reports 256K fine, but meta suggest 128k
  + Or maybe considering remove it?
+ Keep an eye on Strata updates for the new IST-DASLab DeepSeek-V4.1 GGUF layout (llama.cpp's own
  deepseek41 layout, republished 2026-10-27). Strata v0.2.8 still pins the old 4-shard set (re-checked
  2026-10-30), so the DeepSeek lane stays on the old files; convert + repack once upstream's pins move.
  Details and the rules for the lane: notes/strata.md ("20261025 DeepSeek-V4.1-Flash lane (dsv41) first deployment" and "20261030 ISTA's new GGUF for deepseek")
