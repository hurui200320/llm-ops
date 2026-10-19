# Strata (AMD / HIP) deployment

The Strata image for fedora-tuf, built from **upstream's release engine package** — nothing is
compiled on our side. It carries the engine, both picture readers (CPU and GPU) and the ROCm
runtime for the whole RDNA 3 dGPU family (gfx1100 = RX 7900 XTX/XT, gfx1101 = RX 7800/7700 XT,
gfx1102 = RX 7600), the Strata Terminal (serve + the GGUF preparation tools) and the `strata` CLI.
The host needs only the `amdgpu` kernel driver: no ROCm, no compiler.

Built and pushed by [`.github/workflows/strata-image.yml`](../../.github/workflows/strata-image.yml)
(manual trigger) as `ghcr.io/hurui200320/llm-ops-strata:<tag>` — e.g. `0.2.0-beta-rocm-gfx110x`
(public package; pulls need no login).

## Pins (v0.2.0-beta; image verified 2026-10-20)

| What | Value |
| --- | --- |
| Strata app tag | `v0.2.0-beta` (annotated tag → commit `3d1fb48090e8f2cefc22d7b5845fb7f5b71f33fa`) |
| Engine package | `strata-engine-0.1.45.3-linux-x86_64-hip-rx7900.tar.gz`, sha256 `281f660619fe590ee647650729ffcb1108c3f762df5d711123c660ea27ca84fc` |
| Image tag | `0.2.0-beta-rocm-gfx110x` (default; derived from the ref) |
| Pushed digest | `sha256:b6e7f1019e560cc9d6d1a294303ccd8baef37c09ef3841b760c2f32c6bb3a85d` (CI run `39292617432`) — pulled anonymously and smoke-verified on fedora-tuf 2026-10-20; 2.44 GB on disk |
| hipBLASLt table | [`gfx1100-hipblaslt-100500.txt`](gfx1100-hipblaslt-100500.txt) — committed here, byte-identical to the v0.2.0-beta checkout file (sha256 `b314ca1a0f64d80ee1cad7bf85d3483df0e55ae8c5b4c1f78b7067dc5dab2b7a`; CI asserts the equality). Pairing is **runtime-verified at first start**, see below |

Workflow inputs: `strata_ref`, `engine_asset`, `image_tag` (empty = `<ref without v>-rocm-gfx110x`),
`hipblaslt_table_version`. Gates in CI: the release `SHA256SUMS.txt` line, the package manifest
(platform / backend / archs / entrypoint), the gfx1100 table file for the pinned version, and an
in-image `strata components engine` smoke test; upstream's own build stage re-hashes every package
file. The runner needs no GPU.

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
  -e MODEL=IQ3_S -e CONTEXT=256k -e GPUS=0,1 -e KV=fp16 \
  -e MODEL_PATH=/models/strata/ISTA-DASLab-Flash-Next \
  -e STRATA_HIPBLASLT_TUNING=/data/hipblaslt/gfx1100-hipblaslt-100500.txt \
  ghcr.io/hurui200320/llm-ops-strata:0.2.0-beta-rocm-gfx110x
docker logs -f strata-init    # watch the first pass (pack, the one-time MTP fetch)
```

- `MODEL_PATH` takes a folder of GGUF files or one `.gguf` (`GGUF_DIR` still works as an alias).
  Nothing is downloaded for the model itself; the prepared pack and MTP layer are written under
  `/data` (`Strata-data/`).
- No `--memory` cap: 0.2.0 plans from the host's RAM and the entrypoint warns when a cgroup cap is
  seen.
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
  ghcr.io/hurui200320/llm-ops-strata:0.2.0-beta-rocm-gfx110x sh
```

Any command replaces the server start (`docker run … image strata launch --help`, `… image sh`).
Inside: the bundled Python 3.12, `serve/`, the preparation tools the terminal imports — the
`packaging/package_terminal.py` list (`tools/iq_pack.py`, `tools/mtp_fetch.py`, `tools/mtp_pack.py`,
`tools/mtp_rt.py`, `tools/pack_index.py`, `tools/pack_layer.py`, `tools/gguf_reader.py`,
`tools/gguf_writer.py`, `tools/strata_pack.py`, `tools/strata_tokenizer.py`, `tools/strata_mcp.py`,
`tools/canonical_xcheck.py`, `tools/draft_vocab.py`, `tools/_paths.py`) — so engine-side dev tools
such as `quant_quality.py`/`strata_inspect.py` are **not** in the image. Also: llama.cpp's gguf-py
seed (`/opt/strata/seed/third_party/llama.cpp`) and the `strata` CLI (`strata components engine`,
`strata models list`, `strata launch --help/--plan` — all verified on the pulled image 2026-10-20);
`mtp_pack.py`/`mtp_rt.py` called by hand need
`STRATA_GGUF_PY=/opt/strata/seed/third_party/llama.cpp/gguf-py`. There is no compiler in the image:
engine updates are new packages, or a local package built on the host with `packaging/build_hip_linux.sh`
and passed to `docker/build.sh --engine`. `docker exec strata-init strata status` works from outside.

### Local build (fallback)

```sh
git -C <strata-checkout> worktree add /tmp/strata-v0.2.0-beta v0.2.0-beta   # or a fresh clone
tar -xzf strata-engine-0.1.45.3-linux-x86_64-hip-rx7900.tar.gz -C <empty-dir>
/tmp/strata-v0.2.0-beta/docker/build.sh --flavor hip \
  --engine <empty-dir> --repo /tmp/strata-v0.2.0-beta \
  --tag llm-ops-strata:0.2.0-beta-rocm-gfx110x
```

Needs network (base image + the terminal's pinned CPython 3.12 and hash-locked wheels). `--ctx-only`
keeps the assembled context; `STRATA_DOCKER_BUILD_OPTS` adds docker-build options.

**Patch first** — upstream's `docker/build.sh` at v0.2.0-beta (still on main, checked 2026-10-20) stages
`docs/MANAGEMENT_PROTOCOL.md` but not `docs/TERMINAL_LAUNCH.md`, which `packaging/package_terminal.py`'s
PAYLOAD requires, so the build fails with `missing payload item: docs/TERMINAL_LAUNCH.md`. The workflow
appends the missing file automatically (no-op once upstream fixes it); a local build needs the same
one-liner before `docker/build.sh`:

```sh
sed -i 's|docs/MANAGEMENT_PROTOCOL.md|docs/MANAGEMENT_PROTOCOL.md docs/TERMINAL_LAUNCH.md|' <checkout>/docker/build.sh
```

Wiring the image into the host llama-swap (macro with these devices/mounts, per-model env, the
API-key header, `healthCheckTimeout 600`, `docker stop -t 120`) is tracked in the local deployment
plan and lands with the Phase 3 work.

## The hipBLASLt tuning table (why the env var)

The gfx1100 table routes the dense prompt GEMMs through hipBLASLt (docs/AMD_HIP.md: +40-60% prompt
speed on the 7900 XTX); its solution ids are valid only for the hipBLASLt version the engine runs
with. Three verified facts about 0.2.0 (code-checked 2026-10-19):

- the engine reads the table **only** from `STRATA_HIPBLASLT_TUNING` (`src/prefill/gemm.cu`), checks
  magic/arch/version itself and otherwise runs plain hipBLAS;
- the Linux engine package carries no `hipblaslt-version.h` and its `BUILD.json` no
  `hipblaslt_version`; 0.2.0's component flow (`setup.control_engine_step`) resolves neither, so
  `setup.py` never sets the variable on this lane;
- the terminal image ships no `tools/hip/*.txt` files at all.

So the table is wired at run time: it is committed in this repository as
`deploy/strata/gfx1100-hipblaslt-100500.txt` (byte-identical to the `tools/hip/` file of the
`v0.2.0-beta` checkout; the workflow fails if they ever differ). Put it on the host side of the
`/data` bind and pass the env var:

```sh
mkdir -p /mnt/unenc-xfs/strata/hipblaslt
cp /mnt/unenc-xfs/llm-ops/deploy/strata/gfx1100-hipblaslt-100500.txt /mnt/unenc-xfs/strata/hipblaslt/
# run with: -e STRATA_HIPBLASLT_TUNING=/data/hipblaslt/gfx1100-hipblaslt-100500.txt
```

Either env form works (`-e`, or `-e STRATA_SET="STRATA_HIPBLASLT_TUNING=…"` to also save it into the
preset). The engine then logs one of

    prefill gemm: hipBLASLt tuning enabled (32 rows, gfx1100, version 100500)
    prefill gemm: hipBLASLt version mismatch: file=100500 runtime=<N>; using hipBLASEx

`<N>` is the ground truth for this package; pick the matching `gfx1100-hipblaslt-<N>.txt` (or none)
and re-check. This first-start log line is the real pairing gate — CI can only check that a table
for the pinned version exists in the checkout.
