## 20260929

First time deploy, cannot fit everyting in VRAM, have to offload to CPU.

This is a MoE model, so instead of using regular `-ngl` to slice layers,
using `--n-cpu-moe N` to only offload MoE stuff to CPU, so FFN and other
compute-expensive stuff still lives and computes on GPU. This results
in a faster split compared to `-ngl`.

By default (like other models), we're using mmap to map disk blocks to RAM,
so it read as fast as NVME can go. However, when offloading to CPU, the data
structure is not ideal. Using mmap will map GGUF block from disk to RAM,
but llamacpp need to translate that GGUF into something it can compute, add
latency and harm performance. The log will print a warning and recommend using
load mode none. This will make the model loading much slower, but should give
better performance. Need to test this.