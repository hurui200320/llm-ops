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
docker pull ghcr.io/ggml-org/llama.cpp:full-rocm
```

Note that llamacpp's docker image is not the latest, it will build docker image on fixed interval.
If latest docker doesn't work, build our own image:

```bash
docker build \
  -f .devops/rocm.Dockerfile \
  --target full \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-arg APP_VERSION="$(git describe --tags --always)" \
  --build-arg APP_REVISION="$(git rev-parse HEAD)" \
  --progress=plain \
  -t ghcr.io/ggml-org/llama.cpp:full-rocm .
```

### Cache budgets

| Model family | Context per slot | Slots | Checkpoints per slot | Saved-prompt RAM limit |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 26B | 262144 | 2 | 20 | Disabled |
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

## Model startup

The commands disable automatic fitting with `--fit off`: context, GPU placement,
batch sizes and slots are already set explicitly. Inference warmup remains enabled.

All models use `--load-mode dio` for faster cold loads and rotation across the
collection. Unlike `mmap`, direct I/O bypasses the page cache for most weight
reads, so even a previously cached model must be read again. `mmap` was faster
with fully cached weights; it remains an alternative if usage changes to a small
frequently revisited subset. `none` uses buffered reads and was not the best
overall strategy in the switching tests. Context and KV precision are unchanged.

The Docker named volume `llm-ops-comgr-cache` preserves ROCm compilation results
under `/root/.cache/comgr` when serving containers are removed. Docker creates it
automatically. This is a compiler cache, not a model-weight or conversation cache.
Its pruning policy targets 256 MiB and checks at most once per hour; it is not a
hard disk quota. Compiler/version inputs participate in cache keys, so a new
image can compile fresh entries without discarding the old ones first.

Set `AMD_COMGR_CACHE_DIR` explicitly, and use only a policy verified by the runtime.
The tested COMGR build rejects `prune_expiration` despite examples in AMD's docs;
an invalid policy silently disables caching at normal log verbosity. Check for
`llvmcache-*` files in the mounted directory when validating a new image.

Measurements and loading-mode comparisons are recorded in
[notes/model-loading.md](../notes/model-loading.md).

## Security

This config and deployment setup is meant to be used in LAN, more specifically, my home LAN.

So it's wide open with no API keys configured, as I am the only living human in my house.
And I know I will be the only user.
