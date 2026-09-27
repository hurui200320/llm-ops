+ Run benchmarks
+ Add Qwen3.8 27B
  + Considering replace Ornith 1.5 with it
+ Test muse glimmer extend to 256K, community and unsloth reports 256K fine, but meta suggest 128k
  + Or maybe considering remove it?
+ Test unsloth/Laguna-S-2.1-GGUF, reddit said it's pretty good for coding
  + Ask LLM to optimize, maybe off load some layer to CPU without downgrade performance? Like --cpu-moe or --n-cpu-moe?