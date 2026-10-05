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
| Gemma 4 26B | 262144 | 2 | 20 | 40960 MiB |
| Gemma 4 31B QAT | 262144 | 1 | 16 | 40960 MiB |
| Muse Glimmer | 131072 | 4 | 32 | 40960 MiB |
| Ornith 1.5, both variants | 262144 | 1 | 32 | 40960 MiB |
| Qwen3.8, both variants | 262144 | 1 | 16 | 40960 MiB |

Muse disables eager idle-slot RAM copies; its private slots still retain prefixes
in VRAM, and saved-prompt caching remains available when a slot is replaced.
Every family shares one 40960 MiB (40 GiB) saved-prompt budget, large enough to
hold a full-length conversation's saved state (KV plus retained checkpoints):
a 225K-token Qwen3.8 conversation saves about 31 GiB and a full 256K one about
35 GiB, per the `prompt state size` log lines. The budget is an LRU ceiling,
not an allocation: states above it are rejected (logged as "exceeds cache size
limit") and re-prefill on return, and states that do not fit in the 32 GiB RAM
cap may sit in swap until a restore (see "Memory and cache tuning"). Gemma 4's
saved-state sizes were never measured while its cache was disabled, so confirm
the budget holds via the same log lines after a workspace switch. See the
corresponding model notes for measured sizes, latency and the tradeoff for
older-history edits.

## Memory and cache tuning

The host has 64 GB of RAM. The serving containers have a hard 32 GiB RAM cap
plus up to 32 GiB of swap: `--memory-swap` is the *combined* RAM+swap cap, so
`--memory 32g --memory-swap 64g` allows 32 GiB of swap. This is host memory,
not the two GPUs' VRAM. The cap does not make llama-server automatically
shrink its caches: excessive anonymous allocations can still cause a container
OOM kill. Model loading and charged file cache also count toward the RAM cap,
so check the cgroup peak, not just process RSS.

The host keeps a two-tier swap area; the kernel fills it by priority:

| Device | Size | Algorithm | Priority | Role |
| --- | ---: | --- | ---: | --- |
| `/dev/zram0` | 48 GiB | zstd | 100 | First stop: swapped pages compressed in host RAM |
| `/mnt/unenc-xfs/swapfile` | 24 GiB | — | −1 | Cold overflow: disk I/O instead of OOM under a host-wide squeeze |

zram's compressed data still costs host RAM (~1.5–2.5:1 for F16-ish KV), so it
is not free headroom: `zramctl` DATA/COMPR show the live size and the achieved
ratio, and `swapon --show` / `free -h` show what each area actually holds. The
zram area is sized by the `/etc/systemd/zram-generator.conf` override (the
package default in `/usr/lib` is smaller and uses lzo-rle); the swapfile
persists across reboots via its `/etc/fstab` entry.

In the serving containers the active conversation's KV is hot — touched every
decode step — and should stay in RAM; what belongs in swap is the *idle*
conversation's saved-prompt state (the `--cache-ram` pool), touched only on
restore. A swap restore costs a sequential read of the saved state (seconds)
instead of a full re-prefill (minutes). The bad layout is active KV itself
being swapped: `memory.swap.current` climbs *during decode* and tg collapses.
Mitigations, in order: lower `--cache-ram`, enlarge zram, lower `vm.swappiness`
host-wide (60 → 30; affects dev containers too).

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
