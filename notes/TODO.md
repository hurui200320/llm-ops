+ Try `--spec-draft-n-max 3` for Qwen 3.8 27B
+ Test muse glimmer extend to 256K, community and unsloth reports 256K fine, but meta suggest 128k
  + Or maybe considering remove it?
+ Exam strata deployment, tweak docker cli so env and llama-swap is the only config source
  + Strata should pick up those and generate config files everytime on fresh start, no reuse