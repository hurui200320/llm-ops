# Strata (AMD / HIP) deployment

The Strata image for fedora-tuf is **upstream's published image** — nothing is built on our side.
It carries the engine, both picture readers (CPU and GPU) and the ROCm runtime for the whole RDNA 3
dGPU family (gfx1100 = RX 7900 XTX/XT, gfx1101 = RX 7800/7700 XT, gfx1102 = RX 7600), the Strata
Terminal (serve + the GGUF preparation tools) and the `strata` CLI. The host needs only the
`amdgpu` kernel driver: no ROCm, no compiler.

Image: `ghcr.io/niko1221/strata:<version>-rocm-gfx110x` (public package; pulls need no login),
published by upstream's own workflow from the signed release engine packages since 0.2.1
([docs/DOCKER.md](https://github.com/Niko1221/Strata/blob/main/docs/DOCKER.md)). Our own
`strata-image.yml` build was retired on 2026-10-20 after a per-file comparison showed the images
functionally equivalent — engine byte-identical, terminal differing only by the Windows long-path
hotfix that landed after upstream's build. Record: [notes/strata/01-image.md](../../notes/strata/01-image.md).

## Pins (image `0.2.1-rocm-gfx110x`)

| What | Value |
| --- | --- |
| Image | `ghcr.io/niko1221/strata:0.2.1-rocm-gfx110x`, digest `sha256:78551e741f8d47d6edb4b29564cef4e7d33e4f50ab510f3eb1cf0e7adc630fc3` (pulled 2026-10-20) |
| Terminal | upstream commit `c093ac263f8175307e67b7fd7eeeaa2e73ceaded`; `strata --version` = `strata 0.2.1 (build 0.1.46; python 3.12.15 at /opt/strata/terminal/python/bin/python3)` |
| Engine package | `strata-engine-0.1.46-linux-x86_64-hip-rx7900` from the `v0.2.1` release, 241,466,491 bytes, asset sha256 `20e40a4b426c5da8ba0b5979334eb89538e9c4cd51610d456336153e3d45c764`, manifest `build_id a7bdbfdabc9a+rocm`, variant `rx7900`, 580 files — **byte-identical** to the package the retired CI image carried |
| Engine folder in the image | `/opt/strata/components/engine/0.1.46-linux-x86_64-hip-rx7900` (upstream names the folder without the `strata-engine-` prefix; always reference the engine as `component:strata-engine`, never by path) |
| hipBLASLt table | packaged at `tuning/gfx1100-hipblaslt-100200.txt` (sha256 `1bb8cd2e232ce1dec5dc0f303e5bd3b32abd9444330a4428a995ad4db8fb97be`), identical to the retired committed fallback. On gfx1100 it is **opt-in**: pass `STRATA_HIPBLASLT_TABLE=on` as a **container environment variable** (`-e`), **not** inside `STRATA_SET` (see the tuning section) |

First start verified on fedora-tuf (2026-10-21): engine resolves from the official folder, no setup
re-run, `prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)`, `/health` 200,
auth'd `/v1/models` and a short chat answer, clean `docker stop` (3.6 s, exit 0, VRAM freed).

Upstream verifies every package against the signed update feed (Ed25519) and re-hashes every file
inside the build; the gfx110x image itself has not run on a card upstream, but its engine bytes are
the ones runtime-verified on fedora-tuf.

## Upgrading from 0.2.0-beta

- `/data` is unchanged; the existing pack and MTP layer remain valid. **No repack** is needed
  (`iq_pack`, `pack_index`, `pack_layer`, `strata_pack`, `mtp_pack`, `mtp_rt` and the GGUF readers
  did not change between the tags; only `mtp_fetch.py` got faster for new fetches).
- The old `0.2.0-beta` image tag stays available for rollback; switching is an image-tag change.
- Replace the old `STRATA_HIPBLASLT_TUNING=/data/hipblaslt/...` environment with
  **`STRATA_HIPBLASLT_TABLE=on`**. A preset made by 0.2.0 keeps the old key in its `extra`; remove it
  with a backup (`.` env vars are merged, not replaced) or the server logs "set by you" and keeps
  using the old path (same table, but the new packaged-table flow is not tested).
- An installed run config whose `exe` is a path into the old engine folder must be changed to
  `component:strata-engine` (or the model re-run through `strata launch`); the image only carries
  the 0.1.46 component.
- If context/kv/vision are unchanged, the first start is direct (no setup re-run). Do not set
  `KV=fp16`; see the KV bullet below.

### Switching from the retired `ghcr.io/hurui200320/llm-ops-strata` images

Same rules, plus one path caveat:

- `/data` needs **no changes**: the engine package is byte-identical, the manifest and tuning table
  match, presets and packs carry over as-is.
- The engine folder name inside the image differs (`strata-engine-0.1.46-...` → `0.1.46-...`).
  If an installed run config under `/data/install/` names the engine by absolute path, fix it before
  the first start (`grep -r "components/engine" /mnt/unenc-xfs/strata/install/`; a config written by
  `strata launch` on the retired image may hold the old path). Change `exe` to
  `component:strata-engine` (backup first) or re-run the model through `strata launch`.
- `strata --version` reports terminal build `0.1.46` (not `0.1.46.1`), and `strata bench run` records
  the terminal commit `c093ac2…` — keep that in mind when comparing bench records across the switch.

## Run

The first start (or a settings change) downloads/prepares the model; every later start goes straight
to serving. Init/pack passes run outside llama-swap:

```sh
docker run -d --name strata-init \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt label=disable --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
  -v /mnt/unenc-xfs/strata:/data -v /mnt/unenc-xfs/models:/models:z \
  -e MODEL=IQ3_S -e CONTEXT=256k -e GPUS=0,1 -e PARALLEL=1 \
  -e MODEL_PATH=/models/strata/ISTA-DASLab-Flash-Next \
  -e STRATA_HIPBLASLT_TABLE=on \
  -e STRATA_SET="kv=fp16 vram-reserve-mib=3072 vram-reserve-later-mib=700 resident-budget-gib=28 prompt-cache-spread=on conversation-cache-mib=8192 conversation-cache-slots=4 STRATA_SPLIT_OVERLAP=1" \
  ghcr.io/niko1221/strata:0.2.1-rocm-gfx110x
docker logs -f strata-init    # watch the first pass (pack, the one-time MTP fetch)
```

- `MODEL_PATH` takes a folder of GGUF files or one `.gguf` (`GGUF_DIR` still works as an alias).
  Nothing is downloaded for the model itself; the prepared pack and MTP layer are written under
  `/data` (`Strata-data/`).
- **`KV=fp16` still breaks the install** (0.2.1's `setup.py` accepts only `int8|q4_0|k8v4`, while
  `strata launch` passes `--kv fp16` straight through → argparse error → exit 6 with no `[X]` line).
  Install with the default int8 and put **`kv=fp16` in `STRATA_SET`**: lowercase keys become engine
  flags (`--kv fp16` is appended after setup's own `--kv int8` is stripped). Do not set `KV=fp16`
  later either: `desc_matches()` compares it with the installed config and re-runs the broken setup
  on every start.
- **`PARALLEL` still defaults to "fit to the card"** (3 on 24 GB) and `strata launch` turns that into
  a profile override → `--batch 3` (batch slots are otherwise opt-in). Set `PARALLEL=1` for the
  single-request lane (llama-swap `concurrencyLimit: 1`).
- **Session files are unavailable on this split lane, for a different reason than in 0.2.0.**
  `--slot-save-path` is a `strata serve` option, not an engine flag; passing it through `STRATA_SET`
  sends it to the engine and fails with `unknown argument`. Engine 0.1.46 does implement
  `SAVE`/`RESTORE`, but it refuses session files with a layer split
  (`session files do not support --layer-split`; also `--batch`, `--peer-device`). Our production
  pair is a split, so only parking (`--conversation-cache-*`) works. Do not pass the flag.
- No `--memory` cap: 0.2.1 still plans from the host's RAM and the entrypoint warns when a cgroup cap
  is seen.
- `--stop-timeout 120`: SIGTERM drains, unloads and frees the VRAM (`strata launch --stop`). Docker's
  default 10 s is too short for a big model.
- Groups are **gids, not names** here: Docker resolves `--group-add` names against the image's
  `/etc/group` — it has no `render` entry (`--group-add render` fails) and `video` is Ubuntu's 44,
  not the host's — so use the host numbers as above (`getent group video render`; 39/105 on
  fedora-tuf, verified 2026-10-20).
- API key: made on the first start and kept at `/data/state/api-key`; the server listens on the LAN
  (`--access lan`), so a client outside the container must send it as a header (`API_KEY=` /
  `API_KEY_FILE=` override). The production llama-swap lane overrides it with
  `-e API_KEY=${env.STRATA_API_KEY}` (see "llama-swap wiring"), so there the generated key is not
  used.
- Host layout: `/mnt/unenc-xfs/strata/` becomes `/data` — `Strata-data/` (packs, MTP),
  `install/` (run settings, logs), `state/` (presets, key), `config/`, `home/`. Bind mounts only;
  image updates never touch it.

### Toolbox (bash)

```sh
docker run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  -v /mnt/unenc-xfs/strata:/data -v /mnt/unenc-xfs/models:/models:z \
  ghcr.io/niko1221/strata:0.2.1-rocm-gfx110x sh
```

Any command replaces the server start (`docker run … image strata launch --help`, `… image sh`).
Inside: the bundled Python 3.12, `serve/`, the preparation tools the terminal imports — the
`packaging/package_terminal.py` list (`tools/iq_pack.py`, `tools/mtp_fetch.py`, `tools/mtp_pack.py`,
`tools/mtp_rt.py`, `tools/pack_index.py`, `tools/pack_layer.py`, `tools/gguf_reader.py`,
`tools/gguf_writer.py`, `tools/strata_pack.py`, `tools/strata_tokenizer.py`, `tools/strata_mcp.py`,
`tools/canonical_xcheck.py`, `tools/draft_vocab.py`, `tools/_paths.py`) — so engine-side dev tools
such as `quant_quality.py`/`strata_inspect.py` are **not** in the image. Also: llama.cpp's gguf-py
seed (`/opt/strata/seed/third_party/llama.cpp`), `docs/TERMINAL_LAUNCH.md`, the frozen
`bench/prompts/standard-v1` fixtures and the `strata` CLI (`strata components engine`,
`strata models list`, `strata launch --help/--plan`, `strata bench run` — the first four verified
on the pulled 0.2.0 image 2026-10-20; `strata bench run` is new in 0.2.1 and records the terminal's
`COMMIT`). `mtp_pack.py`/`mtp_rt.py` called by hand need
`STRATA_GGUF_PY=/opt/strata/seed/third_party/llama.cpp/gguf-py`. There is no compiler in the image:
engine updates are new packages, or a local package built on the host with `packaging/build_hip_linux.sh`
and passed to `docker/build.sh --engine`. `docker exec strata-init strata status` works from outside.

### Local build (fallback)

Only for a hotfix the official images do not carry yet (a same-day release like `v0.2.1.1`) or a
no-registry host:

```sh
git -C <strata-checkout> worktree add /tmp/strata-v0.2.1.1 v0.2.1.1   # or a fresh clone
tar -xzf strata-engine-0.1.46-linux-x86_64-hip-rx7900.tar.gz -C <empty-dir>
/tmp/strata-v0.2.1.1/docker/build.sh --flavor hip \
  --engine <empty-dir> --repo /tmp/strata-v0.2.1.1 \
  --version 0.2.1 --tag strata:0.2.1-rocm-gfx110x-local
```

Needs network (base image + the terminal's pinned CPython 3.12 and hash-locked wheels). `--ctx-only`
keeps the assembled context; `STRATA_DOCKER_BUILD_OPTS` adds docker-build options.

Build from **`v0.2.1.1` or later**, not `v0.2.1`: the 0.2.1 tag's `package_terminal.py` calls `git`
inside the terminal build stage, which has no git, so the image build fails. `v0.2.1.1` passes
`STRATA_COMMIT` into the build. The old `docs/TERMINAL_LAUNCH.md` patch is no longer needed;
0.2.1+ stages it in `docker/build.sh`.

## llama-swap wiring (Phase 3)

The host llama-swap ([deploy/llama-swap.config.yaml](../llama-swap.config.yaml)) starts this image
with the macro **`docker-server-strata-rocm`** and one model, **`qwen3.8-flash-next-iq3s`**.
Implementation record and the verification checklist: [notes/strata/03-llama-swap.md](../../notes/strata/03-llama-swap.md).

- **No memory cap.** Upstream advises none and the entrypoint warns on `memory.max`; the engine
  plans from the host's `/proc/meminfo` and the 28 GiB resident budget is the real control
  (Phase 2 ran uncapped). Measure `memory.peak` in Phase 5 before considering a cap.
- **Bind mounts only**, same tree as the manual recipe (`/data`, `/models`). No named volumes and no
  comgr cache: the engine is AOT-built. `/templates` is not mounted — Strata never reads it (a chat
  template override is a file swap inside the pack).
- **Gids, not names**: Docker resolves `--group-add` names against the image's `/etc/group` (no
  `render`, `video` is 44) — the macro passes the host numbers (39/105 on fedora-tuf).
- **Loopback publish**: `-p 127.0.0.1:${PORT}:8080`. Inside the container the server still binds
  `0.0.0.0` (the entrypoint's `--access lan`), which is what makes the published port work.
- **One API key, two layers.** The container's server always requires a key beyond loopback
  (`serve/server.py`; the entrypoint binds `0.0.0.0`), so the macro passes
  `-e API_KEY=${env.STRATA_API_KEY}` and llama-swap's `apiKeys` is the same `STRATA_API_KEY`.
  llama-swap validates the client and its reverse proxy forwards the client's
  `Authorization`/`x-api-key` verbatim, so the container sees the same credential. It cannot inject
  an upstream key on its own: only `peers.*.apiKey` does that, local models have no header field.
  Consequences: every client needs the key; Basic auth works for llama.cpp models but not for Strata
  (Bearer/x-api-key only); llama-swap's capability probe sends no auth, so the model entry declares
  `capabilities` instead of relying on discovery.
- **Per-model knobs**: `healthCheckTimeout: 600` (cold start 1-3 min; llama-swap also bounds the
  graceful stop by this value, so `cmdStop: docker stop -t 120 ${MODEL_ID}` fits), `concurrencyLimit:
  1` (matches `PARALLEL=1`), `checkEndpoint: /health` (answers without a key).
- **Env passthrough is the Phase-2 recipe verbatim**: `MODEL`, `CONTEXT=256k`, `GPUS=0,1`,
  `PARALLEL=1`, `MODEL_PATH`, and the `STRATA_SET` string (with `kv=fp16`); plus the container env
  `STRATA_HIPBLASLT_TABLE=on`. Never `KV=`, never `--slot-save-path`.
- The ORCA (uncensored) model entry is Phase 4 work: the pack does not exist yet, so no entry is
  staged for it. Parking knobs already ride in `STRATA_SET`; per-model settings wait for the
  HIP+split parking check (Phase 5).

## The hipBLASLt tuning table (packaged since 0.2.1)

The gfx1100 table routes the dense prompt GEMMs through hipBLASLt (docs/AMD_HIP.md: +40-60% prompt
speed on the 7900 XTX); its solution ids are valid only for the hipBLASLt version the engine runs
with. Since 0.2.1 the AMD engine package carries `tuning/gfx1100-hipblaslt-100200.txt` (listed and
hashed in `engine-component.json`) and records `hipblaslt_version: 100200` in `BUILD.json`. The
server resolves the table for the card at start (`serve/server.py` with
`strata_control/hipblaslt_tuning.py`) and passes it to the engine; the engine still reads only
`STRATA_HIPBLASLT_TUNING` and validates magic/arch/version itself.

- On gfx1100 the packaged table is **opt-in**: set `STRATA_HIPBLASLT_TABLE=on`. Without it the start
  logs `... is in this package and not used by default (faster prompts, but answers can differ in
  the last digits)` and the prompt GEMMs run plain hipBLAS.
  **The switch is read from the server process's environment** (`strata_control/hipblaslt_tuning.py`
  `mode()`: `os.environ` first, then a top-level `"hipblaslt_table"` key of the run config). So it
  must be a container env var (`-e STRATA_HIPBLASLT_TABLE=on`); putting it inside `STRATA_SET`
  silently does nothing — the key is saved into the preset's `env` and only forwarded to the engine,
  which reads `STRATA_HIPBLASLT_TUNING` (a path), not the switch. Verified both ways on fedora-tuf
  2026-10-21: `STRATA_SET` placement → no table; container env →
  `prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)`.
- `STRATA_HIPBLASLT_TABLE=off` never uses a table; `auto` uses it by default only on gfx1151.
- A `STRATA_HIPBLASLT_TUNING` you set yourself always wins (the same table is fine); this remains the
  fallback for the 0.2.0 image and for a package whose table is missing. The retired committed
  copies in this repo were removed with the build; the package is the only copy the image needs. If
  a host-side fallback file is ever wanted, copy it out of the image:
  `docker run --rm IMAGE sh -c 'cat /opt/strata/components/engine/*/tuning/gfx1100-hipblaslt-100200.txt'`.
  The `/mnt/unenc-xfs/strata/hipblaslt/` copy is no longer needed by 0.2.1.
- Start log lines to check:

      hipBLASLt tuning table: gfx1100-hipblaslt-100200.txt (faster prompts; STRATA_HIPBLASLT_TABLE=off turns it off)
      prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)

  A mismatch (`version mismatch: file=100200 runtime=<N>`) means the package's `BUILD.json`/table
  pairing is wrong; pick the matching `gfx1100-hipblaslt-<N>.txt` (or none) and re-check.
