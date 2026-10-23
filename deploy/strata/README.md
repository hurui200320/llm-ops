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
hotfix that landed after upstream's build. Record: [notes/strata.md](../../notes/strata.md).

## Pins (image `0.2.2-rocm-gfx110x`)

| What | Value |
| --- | --- |
| Image | `ghcr.io/niko1221/strata:0.2.2-rocm-gfx110x`, digest `sha256:99ae1768abd51d14ad684f751f42a256ad5f41079adf3ec5da04673c659ebc5b` (pulled 2026-10-21), 1,842,812,422 B / 1.84 GB on disk |
| Terminal | upstream tag `v0.2.2`, commit `e6fff4ad12908de74952abe53b2bade45a68aa0f`; `strata --version` = `strata 0.2.2 (build 0.1.47; python 3.12.15 at /opt/strata/terminal/python/bin/python3)`; image entrypoint sha256 `7c74cfd3…` (byte-identical to the `v0.2.2` checkout) |
| Engine package | `strata-engine-0.1.47-linux-x86_64-hip-rx7900` from the `v0.2.2` release, 242,020,538 bytes, asset sha256 `ecaaee56a0fb44313ed246f57655f95857f017a64cb5fb43104b0417c86b21ba`, manifest `build_id 76f90e58436d+rocm`, variant `rx7900`, 581 files — **byte-identical** to the package in the image (`engine-component.json` sha256 matches) |
| Engine folder in the image | `/opt/strata/components/engine/0.1.47-linux-x86_64-hip-rx7900` (upstream names the folder without the `strata-engine-` prefix; always reference the engine as `component:strata-engine`, never by path) |
| hipBLASLt table | packaged at `tuning/gfx1100-hipblaslt-100200.txt` (sha256 `1bb8cd2e232ce1dec5dc0f303e5bd3b32abd9444330a4428a995ad4db8fb97be`), **unchanged from 0.2.1**. On gfx1100 it is **opt-in**: pass `STRATA_HIPBLASLT_TABLE=on` as a **container environment variable** (`-e`), **not** inside `STRATA_SET` (see the tuning section) |

0.2.2 first start verified on fedora-tuf (2026-10-21) on a throwaway copy of the vanilla data tree:
the entrypoint drops the 0.2.1 install record (it names the 0.1.46 vision reader), setup re-runs
once with **no repack**, and the lane serves keyless on a loopback-published bridge port. Engine
0.1.47 gave the same split (`K=22`, `CUDA0` then `CUDA1`), 26.81 GiB resident, 7332 + 6875 expert
slots, `prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)`, `/health` 200,
`/v1/models` 200 without a key, a short chat answer, and a clean `docker stop` (2.9 s, exit 0,
VRAM freed). The SC117 lane passed the same start; its setup picks the lane's own mmproj (see the
wiring section). Deployed on fedora-tuf 2026-10-21; both lanes serve. On SC117 the keyless
bridge/loopback, `:npt`, a declared-tool call, 640×360 vision with the lane's own mmproj and the
`/upstream/` UI assets were verified. Record: [notes/strata.md](../../notes/strata.md).

Upstream verifies every package against the signed update feed (Ed25519) and re-hashes every file
inside the build; the gfx110x image itself has not run on a card upstream, but its engine bytes are
the ones runtime-verified on fedora-tuf.

## Upgrading from 0.2.1 (0.2.2, engine 0.1.47)

- `/data` needs **no repack**: `iq_pack`, `pack_index`, `pack_layer`, `strata_pack`, `mtp_pack`,
  `mtp_rt` and the GGUF readers are unchanged between `v0.2.1.4` and `v0.2.2`. The existing pack
  and MTP layer load as-is.
- The 0.2.2 entrypoint drops a run config whose `vision` fields name a missing engine folder. Both
  lanes therefore do **one setup pass** on the first 0.2.2 start; the log says
  `The saved run settings strata-iq3_s.json name /opt/strata/components/engine/0.1.46-... ... making
  them again. The model files are kept.` The pack is not rebuilt; a warm first start took ~30 s in
  the 2026-10-21 smoke.
- **SC117:** setup picks the mmproj from the lane's own model folder, so the lane now uses
  `sc117-Flash-Next-uncensored/mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,542,944 B, sha256
  `2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355`) instead of the pinned ISTA
  file (`b1a82259…`). This is a deliberate choice: the copy is the lane's own file and matches its
  sidecar; it differs from the ISTA pin only in a 64-byte header (334 tensors identical). The 0.2.2
  smoke started and served with it, but did not exercise vision; keep that smoke on the list.
- **`--no-api-key`:** the flag is in Terminal 0.1.47. The official 0.2.1 image bundles 0.1.46,
  which does not have it, so 0.2.2 is the first official image that can run the lane keyless on a
  non-loopback bind. `deploy/strata/entrypoint-local.sh` adds the flag only when no key is set.
- The engine is 0.1.47. The gfx1100 hipBLASLt table is unchanged (same file, hash and version
  `100200`). The release notes claim no RX 7000 speed change.
- **Rollback:** back up `install/strata-iq3_s.json` and
  `config/strata/launch-presets/docker.json` before the first start. 0.2.2 rewrites the install
  record with the 0.1.47 vision paths; the 0.2.1 entrypoint does not repair a config that names a
  missing component, so a rollback to 0.2.1 needs those backups. The pack and MTP are compatible
  both ways.
- **Deploy order:** pull this checkout on fedora-tuf, confirm
  `deploy/strata/entrypoint-local.sh` is mode 755, make sure no Strata model is loaded, then let
  llama-swap's `-watch-config` reload (or restart it). Start the vanilla lane first and watch its
  setup pass. The older 0.2.1 image stays on the host for rollback.

## Upgrading from 0.2.0-beta (historical)

- `/data` is unchanged; the existing pack and MTP layer remain valid. **No repack** is needed
  (`iq_pack`, `pack_index`, `pack_layer`, `strata_pack`, `mtp_pack`, `mtp_rt` and the GGUF readers
  did not change between the tags; only `mtp_fetch.py` got faster for new fetches).
- The old `0.2.0-beta` image tag stays available for rollback; switching is an image-tag change.
- Replace the old `STRATA_HIPBLASLT_TUNING=/data/hipblaslt/...` environment with
  **`STRATA_HIPBLASLT_TABLE=on`**. A preset made by 0.2.0 keeps the old key in its `extra`; remove it
  with a backup (`.` env vars are merged, not replaced) or the server logs "set by you" and keeps
  using the old path (same table, but the new packaged-table flow is not tested).
- An installed run config whose `exe` is a path into the old engine folder must be changed to
  `component:strata-engine` (or the model re-run through `strata launch`); the 0.2.1 image only
  carried the 0.1.46 component.
- In the 0.2.0→0.2.1 switch, if context/kv/vision were unchanged, the first start was direct (no
  setup re-run). This does not apply to 0.2.2; see the 0.2.1→0.2.2 section above. Do not set
  `KV=fp16`; see the KV bullet below.

### Switching from the retired `ghcr.io/hurui200320/llm-ops-strata` images

Same rules, plus one path caveat:

- `/data` needs **no changes**: the engine package is byte-identical, the manifest and tuning table
  match, presets and packs carry over as-is.
- The engine folder name inside the image differs (`strata-engine-0.1.46-...` → `0.1.46-...`).
  If an installed run config under `/data/install/` names the engine by absolute path, fix it before
  the first start (`grep -r "components/engine" /mnt/unenc-xfs/strata/ISTA-DASLab-Flash-Next/install/`; a config written by
  `strata launch` on the retired image may hold the old path). Change `exe` to
  `component:strata-engine` (backup first) or re-run the model through `strata launch`.
- `0.2.1` reported terminal build `0.1.46` (not `0.1.46.1`), and `strata bench run` records the
  terminal commit `c093ac2…` — keep that in mind when comparing bench records across the switch.

## Run

The first start (or a settings change) downloads/prepares the model; every later start goes straight
to serving. Init/pack passes run outside llama-swap:

```sh
docker run -d --name strata-init --init --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt label=disable --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
  -p 127.0.0.1:8080:8080 \
  --entrypoint /entry-local.sh \
  -e PORT=8080 \
  -v /mnt/unenc-xfs/llm-ops/deploy/strata/entrypoint-local.sh:/entry-local.sh:ro \
  -v /mnt/unenc-xfs/strata/ISTA-DASLab-Flash-Next:/data -v /mnt/unenc-xfs/models:/models:z \
  -e MODEL=IQ3_S -e CONTEXT=256k -e GPUS=0,1 -e PARALLEL=1 \
  -e MODEL_PATH=/models/strata/ISTA-DASLab-Flash-Next \
  -e STRATA_HIPBLASLT_TABLE=on \
  -e STRATA_SET="kv=fp16 vram-reserve-mib=1536 vram-reserve-later-mib=768 resident-budget-gib=28 prompt-cache-spread=on conversation-cache-mib=8192 conversation-cache-slots=4 pipeline-windows=2 layer-split=26 trim-stage-weights=on spec-min-p=0.7 STRATA_SCORES_THREAD=1" \
  ghcr.io/niko1221/strata:0.2.2-rocm-gfx110x
docker logs -f strata-init    # 0.2.2 re-runs setup once; the pack and MTP are reused
```

This uses the same keyless entrypoint as the swap lanes, so an init run does not create an API key.
Stop the container when setup finishes; the swap lane then reuses the prepared tree. The current
lane settings are also listed in [notes/strata.md](../../notes/strata.md).

- `MODEL_PATH` takes a folder of GGUF files or one `.gguf` (`GGUF_DIR` still works as an alias).
  Nothing is downloaded for the model itself; the prepared pack and MTP layer are written under
  `/data` (`Strata-data/`).
- **`KV=fp16` still breaks the install** (0.2.2's `setup.py` accepts only `int8|q4_0|k8v4`, while
  `strata launch` passes `--kv fp16` straight through → argparse error → exit 6 with no `[X]` line).
  Install with the default int8 and put **`kv=fp16` in `STRATA_SET`**: lowercase keys become engine
  flags (`--kv fp16` is appended after setup's own `--kv int8` is stripped). Do not set `KV=fp16`
  later either: `desc_matches()` compares it with the installed config and re-runs the broken setup
  on every start.
- **`PARALLEL` still defaults to "fit to the card"** (3 on 24 GB) and `strata launch` turns that into
  a profile override → `--batch 3` (batch slots are otherwise opt-in). Set `PARALLEL=1` for the
  single-request lane (the swap's `concurrencyLimit: 20` is only its queue depth).
- **Session files are unavailable on this split lane, for a different reason than in 0.2.0.**
  `--slot-save-path` is a `strata serve` option, not an engine flag; passing it through `STRATA_SET`
  sends it to the engine and fails with `unknown argument`. Engine 0.1.47 does implement
  `SAVE`/`RESTORE`, but it refuses session files with a layer split
  (`session files do not support --layer-split`; also `--batch`, `--peer-device`). Our production
  pair is a split, so only parking (`--conversation-cache-*`) works. Do not pass the flag.
- No `--memory` cap: 0.2.2 still plans from the host's RAM and the entrypoint warns when a cgroup cap
  is seen.
- `--stop-timeout 120`: SIGTERM drains, unloads and frees the VRAM (`strata launch --stop`). Docker's
  default 10 s is too short for a big model.
- Groups are **gids, not names** here: Docker resolves `--group-add` names against the image's
  `/etc/group` — it has no `render` entry (`--group-add render` fails) and `video` is Ubuntu's 44,
  not the host's — so use the host numbers as above (`getent group video render`; 39/105 on
  fedora-tuf, verified 2026-10-20).
- API key: with the upstream entrypoint, a manual `docker run` uses the LAN bind (`--access lan`)
  and, without `API_KEY` or `--no-api-key`, makes a key on the first start and keeps it at
  `/data/state/api-key`; a client outside the container must send it (`API_KEY=` /
  `API_KEY_FILE=` override). The llama-swap lanes and the init command above use the fork instead:
  it adds `--no-api-key`, writes no key, the container runs on a bridge network, and only
  `127.0.0.1:${PORT}` is published on the host (see "llama-swap wiring").
- Host layout: each lane is its own data tree, mounted at `/data`:
  `/mnt/unenc-xfs/strata/ISTA-DASLab-Flash-Next/` (vanilla) or
  `/mnt/unenc-xfs/strata/sc117-Flash-Next-uncensored/` (SC117). Each has `Strata-data/` (packs,
  MTP), `install/` (run settings, logs), `state/` (presets, key), `config/`, `home/`. The GGUFs
  stay shared under `/mnt/unenc-xfs/models/`. Bind mounts only; image updates never touch them.

### Toolbox (bash)

```sh
docker run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  -v /mnt/unenc-xfs/strata/ISTA-DASLab-Flash-Next:/data -v /mnt/unenc-xfs/models:/models:z \
  ghcr.io/niko1221/strata:0.2.2-rocm-gfx110x sh
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

Only for a hotfix the official images do not carry yet (a same-day release like `v0.2.2`) or a
no-registry host:

```sh
git -C <strata-checkout> worktree add /tmp/strata-v0.2.2 v0.2.2   # or a fresh clone
tar -xzf strata-engine-0.1.47-linux-x86_64-hip-rx7900.tar.gz -C <empty-dir>
/tmp/strata-v0.2.2/docker/build.sh --flavor hip \
  --engine <empty-dir> --repo /tmp/strata-v0.2.2 \
  --version 0.2.2 --tag strata:0.2.2-rocm-gfx110x-local
```

Needs network (base image + the terminal's pinned CPython 3.12 and hash-locked wheels). `--ctx-only`
keeps the assembled context; `STRATA_DOCKER_BUILD_OPTS` adds docker-build options.

Build from **`v0.2.1.1` or later** (0.2.2 is fine): the 0.2.1 tag's `package_terminal.py` calls
`git` inside the terminal build stage, which has no git, so the image build fails. `v0.2.1.1`
passes `STRATA_COMMIT` into the build. The old `docs/TERMINAL_LAUNCH.md` patch is no longer needed;
0.2.1+ stages it in `docker/build.sh`.

## llama-swap wiring

The host llama-swap ([deploy/llama-swap.config.yaml](../llama-swap.config.yaml)) starts this image
with the macro **`docker-server-strata-rocm`** and two models appended at the end of `models:` —
vanilla **`qwen3.8-flash-next-iq3s`** and SC117 **`qwen3.8-flash-next-sc117-iq3s`** (`${PORT}` is
assigned in config order: 10009 / 10010).
Implementation record and current pins:
[notes/strata.md](../../notes/strata.md).

- **No memory cap.** Upstream advises none and the entrypoint warns on `memory.max`; the engine
  plans from the host's `/proc/meminfo` and the 28 GiB resident budget is the real control. The
  lane runs uncapped; measure `memory.peak` if a cap is ever considered.
- **Bind mounts only**, same tree as the manual recipe (`/data`, `/models`). No named volumes and no
  comgr cache: the engine is AOT-built. `/templates` is not mounted — Strata never reads it (a chat
  template override is a file swap inside the pack).
- **Gids, not names**: Docker resolves `--group-add` names against the image's `/etc/group` (no
  `render`, `video` is 44) — the macro passes the host numbers (39/105 on fedora-tuf).
- **Keyless, on a Docker bridge.** Strata 0.2.2 makes the API key optional; upstream's entrypoint
  still hardcodes `--access lan` (0.0.0.0). This lane runs on the default Docker bridge with
  [entrypoint-local.sh](entrypoint-local.sh), a fork of upstream's entrypoint that passes
  `--no-api-key` when no key is set and suppresses upstream's false "Strata made one" warning. Docker
  publishes only `127.0.0.1:${PORT}` (`-p 127.0.0.1:${PORT}:${PORT}`) on the host, where llama-swap
  proxies; there is no `--network host`. `launch` stores no key, so llama-swap stays `apiKeys: []`
  and clients send no header; the LAN reaches Strata only through llama-swap. The browser UI works
  at `http://<fedora-tuf>:8080/upstream/qwen3.8-flash-next-iq3s/` (the app uses relative asset and
  API paths, which llama-swap's prefix stripping preserves).
- **Re-diff the entrypoint fork on every image bump.** Extract the new
  `/opt/strata/entrypoint.sh` from the image
  (`docker run --rm --entrypoint /bin/sh IMAGE -c 'cat /opt/strata/entrypoint.sh'`) and compare it
  with `entrypoint-local.sh`; re-apply the two changes (the `--no-api-key` conditional and the final
  keyless warning) and nothing else. The fork must exist in the fedora-tuf checkout before the first
  start (mode 755): a missing bind source is created by Docker as a root-owned directory and the
  container dies on `--entrypoint`.
- **Preset/key state (2026-10-21).** 0.2.2's `--no-api-key` rewrites each lane's `docker` preset to
  `"access": "lan"`, `"no_key": true`, `"api_key": null`; the private launch config gets
  `"api_key": "none"` (the server's "no key" sentinel), and no `/data/state/api-key` is written.
  The 0.2.1 hand edit (`"access": "local"`, `"api_key": null`, backup
  `docker.json.bak-20261021-pre-local`) is superseded; the retired `api-key.pre-local.bak` can stay.
  A manual run with `API_KEY` still creates and uses a key.
- **Per-model knobs**: `healthCheckTimeout: 600` (cold start 1-3 min; llama-swap also bounds the
  graceful stop by this value, so `cmdStop: docker stop -t 120 ${MODEL_ID}` fits), `concurrencyLimit:
  20` (queue depth; the engine still has one slot — raised from 1 so concurrent resource requests do
  not get 429), `checkEndpoint: /health` (answers without a key); `capabilities` is
  declared, not probed. llama-swap selects its prober from `/v1/models`' `owned_by`, which Strata
  omits, so discovery would fall back to the listing-only prober (context + text; no tools/image).
  Strata's `/props` is llama.cpp-shaped (`chat_template_caps`, `modalities`, measured 2026-10-21),
  so if upstream ever reports `owned_by: llamacpp` the probe becomes complete.
- **Env passthrough is the production recipe in [notes/strata.md](../../notes/strata.md)**:
  `MODEL`, `CONTEXT=256k`, `GPUS=0,1`,
  `PARALLEL=1`, `MODEL_PATH`, and the `STRATA_SET` string (with `kv=fp16`); plus the container env
  `STRATA_HIPBLASLT_TABLE=on`. Never `KV=`, never `--slot-save-path`. Neither entry passes
  `PRESET`: the image default `docker` is saved per lane under each tree's
  `config/strata/launch-presets/`.
- The uncensored lane is **SC117 abliterated IQ3_S** (not ORCA). Its entry
  **`qwen3.8-flash-next-sc117-iq3s`** landed 2026-10-21 as an exact copy of the vanilla block with
  only the SC117 `/data` mount and `MODEL_PATH` changed. The SC117 data tree
  `/mnt/unenc-xfs/strata/sc117-Flash-Next-uncensored/` has the built `packs/iq3_s`, a copied
  `mtp/rt`, and an install record. 0.2.2's first start drops that record (it names the 0.1.46
  vision reader) and re-runs setup; it selects the lane's own mmproj
  (`907,542,944 B`, sha256 `2e788f8c…`), not the ISTA pin. Vision with this copy was verified with
  a 640×360 image after deployment. Despite the unchanged GSQ-RCO layout the ISTA pack cannot be
  reused (95 tensors changed type, 1,214/1,223 offsets moved). Details: `notes/strata.md`.
  Parking knobs already ride in `STRATA_SET`; current per-lane settings are in `notes/strata.md`.

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
  The `/mnt/unenc-xfs/strata/ISTA-DASLab-Flash-Next/hipblaslt/` copy is no longer needed by 0.2.1+.
- Start log lines to check:

      hipBLASLt tuning table: gfx1100-hipblaslt-100200.txt (faster prompts; STRATA_HIPBLASLT_TABLE=off turns it off)
      prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)

  A mismatch (`version mismatch: file=100200 runtime=<N>`) means the package's `BUILD.json`/table
  pairing is wrong; pick the matching `gfx1100-hipblaslt-<N>.txt` (or none) and re-check.
