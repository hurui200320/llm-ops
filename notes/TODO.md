+ Try `--spec-draft-n-max 3` for Qwen 3.8 27B
+ Test muse glimmer extend to 256K, community and unsloth reports 256K fine, but meta suggest 128k
  + Or maybe considering remove it?
+ Test 512k context for Qwen3.8 Flash Next and see speed drop
+ Skill for tunning speed?
  + First collect what knobs are available to us, then get a baseline of current settings,
  + then trying to optimize for pp and tg speed, or subject provided by user
  + during tunning, record to notes after testing each knob, to prevent context compaction throw previous result without recoding it
  + record both working and not working things (extract examples from existing notes?)