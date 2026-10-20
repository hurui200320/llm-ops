# Strata (AMD / HIP) deployment

The Strata image for fedora-tuf, built from **upstream's release engine package** — nothing is
compiled on our side. It carries the engine, both picture readers (CPU and GPU) and the ROCm
runtime for the whole RDNA 3 dGPU family (gfx1100 = RX 7900 XTX/XT, gfx1101 = RX 7800/7700 XT,
gfx1102 = RX 7600), the Strata Terminal (serve + the GGUF preparation tools) and the `strata` CLI.
The host needs only the `amdgpu` kernel driver: no ROCm, no compiler.

Built and pushed by [`.github/workflows/strata-image.yml`](../../.github/workflows/strata-image.yml)
(manual trigger) as `ghcr.io/hurui200320/llm-ops-strata:<tag>` — e.g. `0.2.1-rocm-gfx110x`
(public package; pulls need no login).

## Pins (0.2.1 sources / 0.2.1 release; image `0.2.1-rocm-gfx110x`)

| What | Value |
| --- | --- |
| Strata source tag | `v0.2.1.1` (annotated tag → commit `e55d063114d3cd9220521e6f0a80b61950ab5889`) — the Terminal/Docker hotfix; the release assets are under `v0.2.1` |
| Release / engine assets | `v0.2.1` |
| Engine package | `strata-engine-0.1.46-linux-x86_64-hip-rx7900.tar.gz`, 241,466,491 bytes, sha256 `20e40a4b426c5da8ba0b5979334eb89538e9c4cd51610d456336153e3d45c764`; manifest schema 2, `build_id a7bdbfdabc9a+rocm`, variant `rx7900`, 580 files |
| Image tag | `0.2.1-rocm-gfx110x` (the workflow default; sources are `v0.2.1.1`) |
| Local validation build | `llm-ops-strata:0.2.1-rocm-gfx110x`, 1,435,653,450 bytes, image ID `sha256:51316f1c5a80ec1fe0f4b24b04118310a02e364b73d401361dd112c88458a3d1` — `strata 0.2.1 (build 0.1.46.1)`, engine inventory usable (2026-10-20, GPU-less host); not pushed |
| Pushed digest | _pending the first CI run of this workflow revision_ — record the run ID, digest and size here |
| Terminal build | `0.1.46.1` (from the `v0.2.1.1` CMake version; `strata --version` reports app `0.2.1`, build `0.1.46.1`) |
| hipBLASLt table | packaged at `tuning/gfx1100-hipblaslt-100200.txt` (sha256 `1bb8cd2e232ce1dec5dc0f303e5bd3b32abd9444330a4428a995ad4db8fb97be`), byte-identical to [`gfx1100-hipblaslt-100200.txt`](gfx1100-hipblaslt-100200.txt). The start resolves it, but on gfx1100 it is **opt-in**: `STRATA_HIPBLASLT_TABLE=on`. [`gfx1100-hipblaslt-100500.txt`](gfx1100-hipblaslt-100500.txt) stays committed for newer ROCm builds (sha256 `b314ca1a0f64d80ee1cad7bf85d3483df0e55ae8c5b4c1f78b7067dc5dab2b7a`). CI checks the package table, the committed fallback and the checkout copy against each other |

Workflow inputs: `strata_ref` (source tag, default `v0.2.1.1`), `engine_release` (assets, default
`v0.2.1`), `engine_asset`, `image_tag` (default `0.2.1-rocm-gfx110x`), `hipblaslt_table_version`.
Gates in CI: the release `SHA256SUMS.txt` line, the package manifest and `BUILD.json`
(platform / backend / variant / archs / entrypoint / `hipblaslt_version`), the package's gfx1100
table for the pinned version (plus the committed and checkout copies), and an in-image
`strata components engine` smoke test with `strata --version`; upstream's own build stage re-hashes
every package file. The runner needs no GPU.

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
  `component:strata-engine` (or the model re-run through `strata launch`); the new image only carries
  the 0.1.46 component.
- If context/kv/vision are unchanged, the first start is direct (no setup re-run). Do not set
  `KV=fp16`; see the KV bullet below.

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
  -e STRATA_SET="kv=fp16 vram-reserve-mib=3072 vram-reserve-later-mib=700 resident-budget-gib=28 prompt-cache-spread=on conversation-cache-mib=8192 conversation-cache-slots=4 STRATA_SPLIT_OVERLAP=1 STRATA_HIPBLASLT_TABLE=on" \
  ghcr.io/hurui200320/llm-ops-strata:0.2.1-rocm-gfx110x
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
  `API_KEY_FILE=` override).
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
  ghcr.io/hurui200320/llm-ops-strata:0.2.1-rocm-gfx110x sh
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

```sh
git -C <strata-checkout> worktree add /tmp/strata-v0.2.1.1 v0.2.1.1   # or a fresh clone
tar -xzf strata-engine-0.1.46-linux-x86_64-hip-rx7900.tar.gz -C <empty-dir>
/tmp/strata-v0.2.1.1/docker/build.sh --flavor hip \
  --engine <empty-dir> --repo /tmp/strata-v0.2.1.1 \
  --version 0.2.1 --tag llm-ops-strata:0.2.1-rocm-gfx110x
```

Needs network (base image + the terminal's pinned CPython 3.12 and hash-locked wheels). `--ctx-only`
keeps the assembled context; `STRATA_DOCKER_BUILD_OPTS` adds docker-build options.

Build from **`v0.2.1.1` or later**, not `v0.2.1`: the 0.2.1 tag's `package_terminal.py` calls `git`
inside the terminal build stage, which has no git, so the image build fails. `v0.2.1.1` passes
`STRATA_COMMIT` into the build. The old `docs/TERMINAL_LAUNCH.md` patch is no longer needed;
0.2.1+ stages it in `docker/build.sh`.

**Upstream image (cross-check).** `ghcr.io/niko1221/strata:0.2.1-rocm-gfx110x` is public (1.43 GB
on disk when checked 2026-10-20; engine 0.1.46, terminal 0.2.1 build 0.1.46) and is built from a
post-0.2.1 main commit. It is useful to compare the engine inventory and tuning table with ours;
production uses `ghcr.io/hurui200320/llm-ops-strata:0.2.1-rocm-gfx110x`, which carries the
`v0.2.1.1` terminal (build 0.1.46.1) and our package/table gates.

Wiring the image into the host llama-swap (macro with these devices/mounts, per-model env, the
API-key header, `healthCheckTimeout 600`, `docker stop -t 120`) is tracked in the local deployment
plan and lands with the Phase 3 work; this upgrade does not touch the llama-swap config.

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
- `STRATA_HIPBLASLT_TABLE=off` never uses a table; `auto` uses it by default only on gfx1151.
- A `STRATA_HIPBLASLT_TUNING` you set yourself always wins (the same table is fine); this remains the
  fallback for the 0.2.0 image and for a package whose table is missing.
- The committed copy `deploy/strata/gfx1100-hipblaslt-100200.txt` remains for CI and as the host-side
  fallback; the `/mnt/unenc-xfs/strata/hipblaslt/` copy is no longer needed by 0.2.1. The workflow
  fails if the package table, the committed copy and the checkout table ever differ.
- Start log lines to check:

      hipBLASLt tuning table: gfx1100-hipblaslt-100200.txt (faster prompts; STRATA_HIPBLASLT_TABLE=off turns it off)
      prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)

  A mismatch (`version mismatch: file=100200 runtime=<N>`) means the package's `BUILD.json`/table
  pairing is wrong; pick the matching `gfx1100-hipblaslt-<N>.txt` (or none) and re-check. The
  checkout carries gfx1100 tables for `100100/100200/100202/100401/100500`; a mismatch means
  swapping the file, not rebuilding.
