# Deploy

Running on fedora-tuf, which has two AMD 7900XTX.

```
/mnt/unenc-xfs/llama-swap -config /mnt/unenc-xfs/llm-ops/deploy/llama-swap.config.yaml -watch-config
```

+ `/mnt/unenc-xfs/models`: copy the archive/LLM content from NAS
+ `/mnt/unenc-xfs/llm-ops`: git clone of this repo

To download llama-swap (download to /mnt/unenc-xfs/llama-swap):

```
curl -sL https://api.github.com/repos/mostlygeek/llama-swap/releases/latest | jq -r '.assets[] | select(.name | test("linux_amd64")).browser_download_url' | xargs curl -sL | tar xz -C /mnt/unenc-xfs/ llama-swap
chmod +x /mnt/unenc-xfs/llama-swap
```

To update docker images:

```
docker pull ghcr.io/ggml-org/llama.cpp:full-vulkan
docker pull ghcr.io/ggml-org/llama.cpp:full-rocm
```

Note that llamacpp's docker image is not the latest, it will build docker image on fixed interval.
If latest docker doesn't work, build our own image.

### Cache budgets

| Model family | Context per slot | Slots | Checkpoints per slot | Saved-prompt RAM limit |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 26B | 262144 | 1 | 20 | Disabled |
| Gemma 4 31B QAT | 262144 | 1 | 16 | Disabled |
| Muse Glimmer | 131072 | 4 | 32 | 8192 MiB |
| Ornith 1.5, both variants | 262144 | 1 | 32 | 8192 MiB |
| Qwen3.8, both variants | 262144 | 1 | 16 | 4096 MiB |

Muse disables eager idle-slot RAM copies; its private slots still retain prefixes
in VRAM, and saved-prompt caching remains available when a slot is replaced.
Qwen's saved-prompt limit is intentionally bounded: the tested 7.7K-token session
fit, but a 31K-token session plus checkpoints did not. See the corresponding
model notes for measured sizes, latency and the tradeoff for older-history edits.

## Memory and cache tuning

The host has 64 GB of RAM; the serving-container budget is 32 GiB. This is
host memory, not the two GPUs' VRAM. Docker's `--memory 32g --memory-swap 32g`
sets a hard host-memory limit and disables container swap. It does not make
llama-server automatically shrink its caches: excessive anonymous allocations
can still cause a container OOM kill. Model loading and charged file cache also
count toward the limit, so check the cgroup peak, not just process RSS.

For the cgroup-v2 counters, find the container's init PID with
`docker inspect --format '{{.State.Pid}}' <model>`, read its `/proc/<pid>/cgroup`,
and inspect that directory under `/sys/fs/cgroup`. Useful counters are
`memory.current`, `memory.peak`, `memory.stat` (especially `anon` and `file`),
`memory.swap.current`, and `memory.events`. An increasing `max` counter may mean
file-cache reclaim; check `oom` and `oom_kill` separately. `docker stats` alone
does not show the complete cgroup-memory breakdown.

`--cache-ram` only limits saved prompt states. Active context checkpoints have
a separate **per-slot** count limit and can include draft-model state. Minimum
checkpoint spacing is not a promise to create periodic checkpoints across the
entire context. Changing checkpoint counts does not change the context window,
but can make edits to older conversation history require more re-prefill.

Keep both target and draft K/V caches at their original F16 precision. Do not
trade coding or agentic-loop reliability for KV-cache quantization. Changes to
host-side caches should be checked with repeated turns, recent-history edits,
and all four Muse slots populated, in addition to the cold-prefill speed test.

Raw synthetic tuning results belong in the gitignored `bench/results/` tree;
dated findings belong in the corresponding model notes. Never record real
conversation prompts or responses in this public repository.

## Security

This config and deployment setup is meant to be used in LAN, more specifically, my home LAN.

So it's wide open with no API keys configured, as I am the only living human in my house.
And I know I will be the only user.
