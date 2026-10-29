+ Try `--spec-draft-n-max 3` for Qwen 3.8 27B
+ Test muse glimmer extend to 256K, community and unsloth reports 256K fine, but meta suggest 128k
  + Or maybe considering remove it?
+ Keep an eye on Strata updates for the new IST-DASLab DeepSeek-V4.1 GGUF layout (llama.cpp's own
  deepseek41 layout, republished 2026-10-27). Strata v0.2.7.1 still pins the old 4-shard set, so the
  DeepSeek lane stays on the old files; convert + repack once upstream's pins move.
  Details and the rules for the lane: notes/strata.md ("DeepSeek-V4.1-Flash lane" -> "Model files and pins")
+ Skill for tunning speed?
  + First collect what knobs are available to us, then get a baseline of current settings,
  + then trying to optimize for pp and tg speed, or subject provided by user
  + during tunning, record to notes after testing each knob, to prevent context compaction throw previous result without recoding it
  + record both working and not working things (extract examples from existing notes?)
