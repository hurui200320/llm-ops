# Strata (AMD / HIP) deployment

The Strata image the lanes run is **our own build**, `ghcr.io/hurui200320/llm-ops-strata:0.2.4.2-rocm-gfx110x`,
made by [`.github/workflows/strata-image.yml`](../../.github/workflows/strata-image.yml). It carries upstream's
Terminal 0.2.4.2 (serve + the GGUF preparation tools and the `strata` CLI), upstream's signed-feed 0.1.48 Qwen engine
package (byte-identical to the 0.2.3 one), both picture readers (CPU and GPU) and the ROCm runtime for the whole RDNA 3
dGPU family (gfx1100 = RX 7900 XTX/XT, gfx1101 = RX 7800/7700 XT, gfx1102 = RX 7600), plus one added file: the
gfx110x `strata-dsv41` build (DeepSeek V4.1-Flash; **not used by these lanes**). The host needs only the `amdgpu`
kernel driver: no ROCm, no compiler.

Upstream's own images come from [docs/DOCKER.md](https://github.com/Niko1221/Strata/blob/main/docs/DOCKER.md) and the
signed release engine packages since 0.2.1. Our first `strata-image.yml` build was retired on 2026-10-20 after a
per-file comparison showed the images functionally equivalent — engine byte-identical, terminal differing only by the
Windows long-path hotfix that landed after upstream's build (record: [notes/strata.md](../../notes/strata.md)). It is
back for 0.2.4.x for two reasons: upstream has published no 0.2.4.x image (its workflow is manual; last run 2026-10-23,
for 0.2.3), and no upstream image can carry a DeepSeek engine on an RX 7900 — the signed feed has `strata-dsv41`
packages for cuda13 and hip-rx9070 only. See "Building the 0.2.4.x image" and the pins below.

## Pins (image `0.2.4.2-rocm-gfx110x`)

| What | Value |
| --- | --- |
| Image | `ghcr.io/hurui200320/llm-ops-strata:0.2.4.2-rocm-gfx110x`, digest `sha256:c67827ee283e5e90be31e6c01925727256a26e76aa5b6ee15f57762fa29cf3bc` (pulled 2026-10-24, CI run 40100350366), `1,466,884,951 B` by `docker inspect .Size`, `amd64/linux`, version label `0.2.4.2`; public package, pulls need no login |
| Terminal | upstream tag `v0.2.4.2`, commit `dfa3f4025c9f1bc4af8cee201b20a2d223061813`; `strata --version` = `strata 0.2.4.2 (build 0.1.49.2; python 3.12.15 at /opt/strata/terminal/python/bin/python3)`; image entrypoint sha256 `7c74cfd3…` (byte-identical to the `v0.2.3` and `v0.2.2` checkouts — the fork's base) |
| Engine package | `strata-engine-0.1.48-linux-x86_64-hip-rx7900` from the `v0.2.4` release (the `v0.2.3` file re-attached: 242,858,823 bytes, asset sha256 `13bcf38c3ebef5261e7a99c6ce67b66a16e2c6a518586d7e96672c0f72178cfa`), manifest `build_id 6f8800aefe27+rocm`, variant `rx7900` — **byte-identical** to the package in the 0.2.3 image: `engine-component.json` sha256 `59a2ac74297b1714314cd30c93008aae4f661e9311b1cd654443e9111b81259b`, `BUILD.json` sha256 `7108976aa906243c2dfaf9d69d0645741cbc11df30a38e377e70ecf6cad564a2`. 580 files listed in the manifest; 582 on disk = 580 + `engine-component.json` + our `strata-dsv41` |
| `strata-dsv41` (added file) | our gfx1100/1101/1102 build from the same `v0.2.4.2` tag (TheRock `7.10.0a20251120`; llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`), placed beside `strata` in the engine folder — the shape of upstream's DSV41=1 rx9070 package. `strata-dsv41 0.1.49.2 build dfa3f4025`, sha256 `53af25677999b847fd7c1e99d72beb54c85955e588301a1b734087430992e9aa`; **not listed in `engine-component.json` by design** (the Terminal accepts binaries beside the engine and the image build `ldd`-checks it). The Qwen lanes never invoke it |
| Engine folder in the image | `/opt/strata/components/engine/0.1.48-linux-x86_64-hip-rx7900` (upstream names the folder without the `strata-engine-` prefix; always reference the engine as `component:strata-engine`, never by path) |
| hipBLASLt table | packaged at `tuning/gfx1100-hipblaslt-100200.txt` (sha256 `1bb8cd2e232ce1dec5dc0f303e5bd3b32abd9444330a4428a995ad4db8fb97be`), **unchanged from 0.2.1/0.2.2/0.2.3** (re-verified in this image). On gfx1100 it is **opt-in**: pass `STRATA_HIPBLASLT_TABLE=on` as a **container environment variable** (`-e`), **not** inside `STRATA_SET` (see the tuning section) |

The image was verified in CI (run 40100350366, which also smoke-tests it in-image) and locally on 2026-10-24 against
the pushed digest: Terminal and component inventory (`strata components engine --backend hip --full` → usable, no
problems, every listed file hashed), `strata-dsv41 --version`, and the entrypoint fork diff (upstream base
byte-identical to `v0.2.3`; exactly the three fork changes, nothing else). Runtime verification on fedora-tuf is
recorded in [notes/strata.md](../../notes/strata.md).

0.2.4.2 keeps the 0.2.3 engine, pack, MTP and install record, so its first start is expected to go
straight to serving — no setup pass, no repack. For reference, the 0.2.3 first start was verified
on fedora-tuf (2026-10-23) on a reflinked throwaway copy of the vanilla data tree: the entrypoint
drops the 0.2.2 install record (it names the 0.1.47 `strata-vision-hip`),
setup re-runs once with **no repack** (50 files in `Strata-data/` before and after; only
`mtp/tensors/verified.json` re-stamped) and **no download**, and the lane serves keyless on a
loopback-published bridge port after about 40 s (≈30 s of it is the model load). Engine 0.1.48
gave the same split (`CUDA0` 0-25 / `CUDA1` 26-47), 22.37 GiB resident, MTP layer 1348 MiB,
`prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)`, `/health` 200,
`/v1/models` 200 without a key, a short chat answer, `max_tokens:-1` accepted, a declared-tool call
(`get_weather{"city":"Paris"}`) and 640×360 vision ("Red"). A warm second start took ≈40 s with no
setup pass, and `docker stop -t 120` took 2.4 s (exit 0, VRAM back to 0.19 GiB). The SC117 lane
uses the same flow; its setup picks the lane's own mmproj (see the wiring section). Record:
[notes/strata.md](../../notes/strata.md).

Upstream verifies every package against the signed update feed (Ed25519) and re-hashes every file
inside the build; the gfx110x image itself has not run on a card upstream, but its engine bytes are
the ones runtime-verified on fedora-tuf.

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
  -v /mnt/unenc-xfs/llm-ops/templates:/templates:ro \
  -v /mnt/unenc-xfs/strata/ISTA-DASLab-Flash-Next:/data -v /mnt/unenc-xfs/models:/models:z \
  -e MODEL=IQ3_S -e CONTEXT=256k -e GPUS=0,1 -e PARALLEL=1 \
  -e MODEL_PATH=/models/strata/ISTA-DASLab-Flash-Next \
  -e STRATA_HIPBLASLT_TABLE=on \
  -e CHAT_TEMPLATE_PATH=/templates/qwen3.8/Flash-Next-chat-template.jinja \
  -e STRATA_SET="kv=fp16 vram-reserve-mib=1536 vram-reserve-later-mib=768 resident-budget-gib=28 prompt-cache-spread=on conversation-cache-mib=8192 conversation-cache-slots=8 pipeline-windows=2 layer-split=26 trim-stage-weights=on spec-min-p=0.7 STRATA_SCORES_THREAD=1" \
  ghcr.io/hurui200320/llm-ops-strata:0.2.4.2-rocm-gfx110x
docker logs -f strata-init    # 0.2.4.2 reuses the 0.2.3 install record; no setup pass expected
```

This uses the same keyless entrypoint as the swap lanes, so an init run does not create an API key.
Stop the container when setup finishes; the swap lane then reuses the prepared tree. A first setup
writes the GGUF's own template (the fork warns when no pack exists yet), and a repack re-extracts it
after the pre-launch swap; either way the next start installs the repo template (see "Chat template
override"). The current lane settings are also listed in [notes/strata.md](../../notes/strata.md).

- `MODEL_PATH` takes a folder of GGUF files or one `.gguf` (`GGUF_DIR` still works as an alias).
  Nothing is downloaded for the model itself; the prepared pack and MTP layer are written under
  `/data` (`Strata-data/`).
- **`KV=fp16` still breaks the install** (0.2.4.2's `setup.py` accepts only `int8|q4_0|k8v4`, while
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
  sends it to the engine and fails with `unknown argument`. Engine 0.1.48 does implement
  `SAVE`/`RESTORE`, but it refuses session files with a layer split
  (`session files do not support --layer-split`; also `--batch`, `--peer-device`). Our production
  pair is a split, so only parking (`--conversation-cache-*`) works. Do not pass the flag.
- No `--memory` cap: 0.2.4.2 still plans from the host's RAM and the entrypoint warns when a cgroup cap
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
  ghcr.io/hurui200320/llm-ops-strata:0.2.4.2-rocm-gfx110x sh
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

## Building the 0.2.4.x image (with the RX 7900 DeepSeek engine)

Run **Actions → strata-image → Run workflow**. [`.github/workflows/strata-image.yml`](../../.github/workflows/strata-image.yml)
is the source of truth: its header says why the image exists and what it changes against an upstream one, its `env:`
block pins the Strata tag, the engine release, the TheRock ROCm line and the hipBLASLt table, and its steps are the
build and its gates (signed fetch, package checks, the `strata-dsv41` build, the in-image smoke tests). The two
remaining switches: `include_dsv41` (add the RX 7900 DeepSeek binary) and `push` (off builds and smoke-tests without
publishing). The run summary carries the digests.

The image fedora-tuf runs was built by run **40100350366** (2026-10-24):
`ghcr.io/hurui200320/llm-ops-strata@sha256:c67827ee283e5e90be31e6c01925727256a26e76aa5b6ee15f57762fa29cf3bc` —
Terminal `v0.2.4.2` (`dfa3f402`), engine `0.1.48` rx7900 (manifest `59a2ac74…`), `strata-dsv41 0.1.49.2 build
dfa3f4025` (`53af2567…`). The 0.2.4 pin of the workflow also produced `0.2.4-rocm-gfx110x`, superseded by 0.2.4.2.

A local build is those same steps run by hand (Docker + network, no GPU): upstream's `docker/fetch-engines.py`,
[build-dsv41.sh](build-dsv41.sh) and `docker/build.sh`. Tags since **v0.2.4.1** also need the nested-`tools/` context
fix the workflow applies; without it the Terminal stage stops at `installer tool missing: tools/dsv41/tune.py`.

A plain upstream image (no DeepSeek engine) still builds (0.2.3 is fine; build from **`v0.2.1.1` or later**, because
0.2.1's `package_terminal.py` calls `git` inside the Terminal build stage, which has no git):

```sh
git -C <strata-checkout> worktree add /tmp/strata-v0.2.3 v0.2.3
tar -xzf strata-engine-0.1.48-linux-x86_64-hip-rx7900.tar.gz -C <empty-dir>
/tmp/strata-v0.2.3/docker/build.sh --flavor hip \
  --engine <empty-dir> --repo /tmp/strata-v0.2.3 \
  --version 0.2.3 --tag strata:0.2.3-rocm-gfx110x-local
```

Needs network (base image + the terminal's pinned CPython 3.12 and hash-locked wheels). `--ctx-only` keeps the
assembled context; `STRATA_DOCKER_BUILD_OPTS` adds docker-build options.

## llama-swap wiring

The host llama-swap ([deploy/llama-swap.config.yaml](../llama-swap.config.yaml)) starts this image
with the macro **`docker-server-strata-rocm`** and two models appended at the end of `models:` —
vanilla **`qwen3.8-flash-next-iq3s`** and SC117 **`qwen3.8-flash-next-sc117-iq3s`** (`${PORT}` is
assigned in config order: 10007 / 10008 — the two lane entries are 7th and 8th in `models:`).
Implementation record and current pins:
[notes/strata.md](../../notes/strata.md).

- **No memory cap.** Upstream advises none and the entrypoint warns on `memory.max`; the engine
  plans from the host's `/proc/meminfo` and the 28 GiB resident budget is the real control. The
  lane runs uncapped; measure `memory.peak` if a cap is ever considered.
- **Bind mounts only**, same tree as the manual recipe (`/data`, `/models`, `/templates`). No named
  volumes and no comgr cache: the engine is AOT-built. `/templates` is the repo's `templates/` checkout,
  read-only: Strata never reads it — the entrypoint fork copies the Qwen template out of it (see "Chat
  template override").
- **Gids, not names**: Docker resolves `--group-add` names against the image's `/etc/group` (no
  `render`, `video` is 44) — the macro passes the host numbers (39/105 on fedora-tuf).
- **Keyless, on a Docker bridge.** Strata 0.2.4.2 still makes the API key optional; upstream's entrypoint
  still hardcodes `--access lan` (0.0.0.0). This lane runs on the default Docker bridge with
  [entrypoint-local.sh](entrypoint-local.sh), a fork of upstream's entrypoint that passes
  `--no-api-key` when no key is set, suppresses upstream's false "Strata made one" warning, and
  installs `CHAT_TEMPLATE_PATH` over every pack's chat template. Docker
  publishes only `127.0.0.1:${PORT}` (`-p 127.0.0.1:${PORT}:${PORT}`) on the host, where llama-swap
  proxies; there is no `--network host`. `launch` stores no key, so llama-swap stays `apiKeys: []`
  and clients send no header; the LAN reaches Strata only through llama-swap. The browser UI works
  at `http://<fedora-tuf>:8080/upstream/qwen3.8-flash-next-iq3s/` (the app uses relative asset and
  API paths, which llama-swap's prefix stripping preserves).
- **Re-diff the entrypoint fork on every image bump.** Extract the new
  `/opt/strata/entrypoint.sh` from the image
  (`docker run --rm --entrypoint /bin/sh IMAGE -c 'cat /opt/strata/entrypoint.sh'`) and compare it
  with `entrypoint-local.sh`; re-apply the three changes (the `--no-api-key` conditional, the final
  keyless warning, and the pack chat-template swap) and nothing else. The fork must exist in the
  fedora-tuf checkout before the first start (mode 755): a missing bind source is created by Docker
  as a root-owned directory and the container dies on `--entrypoint`.
- **Preset/key state.** Each lane's `docker` preset is `"access": "lan"`, `"no_key": true`,
  `"api_key": null`; the private launch config gets `"api_key": "none"` (the server's "no key"
  sentinel), and no `/data/state/api-key` is written. A manual run with `API_KEY` still creates
  and uses a key.
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
  `mtp/rt`, and an install record. 0.2.3's first start drops the 0.2.2 record (it names the 0.1.47
  engine/vision paths) and re-runs setup once; it selects the lane's own mmproj
  (`907,542,944 B`, sha256 `2e788f8c…`), not the ISTA pin. 0.2.4.2 reuses that 0.2.3 record as it
  is (same engine and mmproj), so no setup pass is expected. Vision with this copy was verified with
  a 640×360 image after the 0.2.2 deployment; the 0.2.3 smoke repeated the check on the vanilla
  lane. Despite the unchanged GSQ-RCO layout the ISTA pack cannot be reused (95 tensors changed
  type, 1,214/1,223 offsets moved). Details: `notes/strata.md`.
  Parking knobs already ride in `STRATA_SET`; current per-lane settings are in `notes/strata.md`.

## Chat template override

Strata has no `--chat-template-file` (unlike `llama-server`). Setup extracts the chat template into
the prepared pack at `<data>/Strata-data/packs/<pack>/tokenizer/chat_template.jinja`, and the server
reads that copy at every start. A one-time swap survives re-installs, but a fresh pack or a
pack-format bump re-extracts the GGUF's own template — so both lanes re-apply the repo copy on every
start:

- `entrypoint-local.sh` compares `CHAT_TEMPLATE_PATH` with every pack's copy and replaces it when it
  differs (silent when equal; warns when no pack exists yet).
- The macro mounts the repo's `templates/` at `/templates:ro` — a directory mount, like the llama.cpp
  lanes: a single-file mount would go stale because `fetch-templates.sh` replaces the file by rename —
  and sets `CHAT_TEMPLATE_PATH=/templates/qwen3.8/Flash-Next-chat-template.jinja`.

Updating the template: run `templates/fetch-templates.sh`, commit, `git pull` in
`/mnt/unenc-xfs/llm-ops`, then restart the lane. Nothing is repacked and no GGUF is re-downloaded. A
start that runs setup (fresh pack, repack) serves the GGUF's template once (the fork warns only when
no pack exists yet); the next start installs the repo copy. After a template change, verify a short
chat, a declared tool call and the `:npt` alias (its `chat_template_kwargs.preserve_thinking` must
still be honoured); record the result in [notes/strata.md](../../notes/strata.md).

## The hipBLASLt tuning table (packaged since 0.2.1)

The gfx1100 table routes the dense prompt GEMMs through hipBLASLt (docs/AMD_HIP.md: +40-60% prompt
speed on the 7900 XTX); its solution ids are valid only for the hipBLASLt version the engine runs
with. Since 0.2.1 the AMD engine package carries `tuning/gfx1100-hipblaslt-100200.txt` (listed and
hashed in `engine-component.json`) and records `hipblaslt_version: 100200` in `BUILD.json`. The
server resolves the table for the card at start (`serve/server.py` with
`strata_control/hipblaslt_tuning.py`) and passes it to the engine; the engine validates
magic/arch/version itself and, since 0.2.3, also falls back to the table beside its own exe when
`STRATA_HIPBLASLT_TUNING` is absent (`STRATA_HIPBLASLT_TABLE=off` still turns it off).

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
- A `STRATA_HIPBLASLT_TUNING` you set yourself always wins (the same table is fine); it remains the
  fallback for a package whose table is missing. The package is the only copy the image needs; a
  host-side copy is not used (copy one out of the image if ever wanted):
  `docker run --rm IMAGE sh -c 'cat /opt/strata/components/engine/*/tuning/gfx1100-hipblaslt-100200.txt'`.
- Start log lines to check:

      hipBLASLt tuning table: gfx1100-hipblaslt-100200.txt (faster prompts; STRATA_HIPBLASLT_TABLE=off turns it off)
      prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)

  A mismatch (`version mismatch: file=100200 runtime=<N>`) means the package's `BUILD.json`/table
  pairing is wrong; pick the matching `gfx1100-hipblaslt-<N>.txt` (or none) and re-check.
