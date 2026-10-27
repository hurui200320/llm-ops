# Strata (AMD / HIP) deployment

The Strata image the lanes run is **our own build**, `ghcr.io/hurui200320/llm-ops-strata:0.2.7-rocm-gfx110x`,
made by [`.github/workflows/strata-image.yml`](../../.github/workflows/strata-image.yml). It carries upstream's
Terminal 0.2.7 (serve + the GGUF preparation tools and the `strata` CLI), upstream's signed-feed 0.1.52 Qwen engine
package, both picture readers (CPU and GPU) and the ROCm runtime for the whole RDNA 3
dGPU family (gfx1100 = RX 7900 XTX/XT, gfx1101 = RX 7800/7700 XT, gfx1102 = RX 7600), plus one added file: the
gfx110x `strata-dsv41` build (DeepSeek V4.1-Flash; run by the `deepseek-v4.1-flash-2bpw` lane — see
[notes/strata.md](../../notes/strata.md) — and never by the Qwen lanes). The host needs only the `amdgpu`
kernel driver: no ROCm, no compiler.

Upstream's own images come from [docs/DOCKER.md](https://github.com/Niko1221/Strata/blob/main/docs/DOCKER.md) and the
signed release engine packages since 0.2.1. Since 0.2.5 upstream ships one engine-less installer per OS and has
published no Docker image after 0.2.3. Our first `strata-image.yml` build was retired on 2026-10-20 after a
per-file comparison showed the images functionally equivalent — engine byte-identical, terminal differing only by the
Windows long-path hotfix that landed after upstream's build (record: [notes/strata.md](../../notes/strata.md)). It is
back since 0.2.4 for two reasons: upstream publishes no images, and no upstream image can carry a DeepSeek engine on
an RX 7900 — the signed feed has `strata-dsv41` packages for cuda13 and hip-rx9070 only. See "Building the 0.2.7
image" and the pins below.

## Pins

Current image: `ghcr.io/hurui200320/llm-ops-strata:0.2.7-rocm-gfx110x` (Terminal `v0.2.7`, engine
`0.1.52` rx7900 — the first Qwen engine byte change since 0.1.50 — plus our `strata-dsv41` 0.1.52).
Digests, package hashes, the engine-folder path and the hipBLASLt table hash are in
[notes/strata.md](../../notes/strata.md#pins); the workflow `env:` block pins the build inputs.

- **Engine folder**: `/opt/strata/components/engine/0.1.52-linux-x86_64-hip-rx7900` (upstream names
  the folder without the `strata-engine-` prefix; reference the engine as `component:strata-engine`,
  never by path).
- **`strata-dsv41`** is our gfx110x build from the same `v0.2.7` tag (0.1.52: the prompt-speed stack, the
  0.2.7 defaults and the vision tower), placed beside `strata` in that
  folder (the shape of upstream's DSV41=1 rx9070 package) and deliberately not listed in
  `engine-component.json`. The Qwen lanes never invoke it; the DeepSeek lane runs it through
  `STRATA_DSV41_EXE`, which entrypoint fork change 4 resolves from the package folder (see
  "DeepSeek (dsv41) lane").
- **hipBLASLt table**: packaged at `tuning/gfx1100-hipblaslt-100200.txt`. The 0.2.7 rx7900 package is
  still built with TheRock `7.10.0a20251120` / hipBLASLt 1.2.0 (`docs/AMD_HIP.md` at the tag) and
  ships the **0.1.48 table bytes again** (`1bb8cd2e…`, 845 B) — the 0.1.50 re-tune (`235a26c3…`,
  873 B) is not in it, so prompt numbers can move slightly against the 0.1.50 spot-check before the
  0.1.52 baseline is recorded. On gfx1100 the table is **opt-in**: pass
  `STRATA_HIPBLASLT_TABLE=on` as a **container environment variable** (`-e`), **not** inside `STRATA_SET`
  (see the tuning section).

The image is smoke-tested in CI (fetch gates, the `strata-dsv41` build, the in-image Terminal and
component checks) and verified locally against the pushed digest; the run number, digest and package
hashes are in [notes/strata.md](../../notes/strata.md#pins). Runtime acceptance on fedora-tuf, 0.2.6:
all three lanes kept their install records and served without a setup pass — vanilla and SC117
answered chat and the declared tool call at 262144 ctx (the vanilla lane through llama-swap too,
including the `:npt` alias and `/upstream` path), and the DeepSeek lane ran `strata-dsv41` 0.1.51
with its `36.0 GB` store. The 0.2.5 acceptance (640×360 vision, `max_tokens:-1`, SC117's own mmproj)
and the per-upgrade records are in
[notes/strata.md](../../notes/strata.md#upgrade-history) and
[notes/qwen3.8/Flash-Next.md](../../notes/qwen3.8/Flash-Next.md).

0.2.7 is different from 0.2.5/0.2.6 in one way that shapes the rollout: **every engine package is new
bytes** (0.1.52), so the engine folder changes and each lane drops its install record and runs setup
once — no model download, no repack for the Qwen lanes; the DeepSeek lane's setup pass packs the
vision tower from the on-disk `mmproj-BF16.gguf` (its pinned size/hash already match 0.2.7). The
rollout record is filled into [notes/strata.md](../../notes/strata.md#upgrade-history) as each lane is
verified.

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
  -e VISION_TOKENS=8192 -e VISION_MIN_TOKENS=1024 \
  -e STRATA_SET="kv=fp16 vram-reserve-mib=1536 vram-reserve-later-mib=1024 resident-budget-gib=28 prompt-cache-spread=on conversation-cache-mib=8192 conversation-cache-slots=8 pipeline-windows=2 layer-split=26 trim-stage-weights=on spec-min-p=0.7 STRATA_SCORES_THREAD=1" \
  ghcr.io/hurui200320/llm-ops-strata:0.2.7-rocm-gfx110x
docker logs -f strata-init    # 0.2.7 changes the engine folder (0.1.50 -> 0.1.52): every lane runs its setup once; the Qwen packs are reused (record rewrite only), no repack and no download
```

This uses the same keyless entrypoint as the swap lanes, so an init run does not create an API key.
Stop the container when setup finishes; the swap lane then reuses the prepared tree. A first setup
writes the GGUF's own template (the fork warns when no pack exists yet), and a repack re-extracts it
after the pre-launch swap; either way the next start installs the repo template (see "Chat template
override"). The current lane settings are also listed in [notes/strata.md](../../notes/strata.md).

- `MODEL_PATH` takes a folder of GGUF files or one `.gguf` (`GGUF_DIR` still works as an alias).
  Nothing is downloaded for the model itself; the prepared pack and MTP layer are written under
  `/data` (`Strata-data/`).
- **`kv=fp16` in `STRATA_SET` is still the recipe.** Since 0.2.5, `setup.py` also accepts `--kv fp16`
  (`int8|q4_0|k8v4|fp16`), so a `KV=fp16` no longer exits 6 — but it would re-run setup once (the
  install record carries the KV choice) and rewrite the lane's record. The lanes stay installed with
  int8; **`kv=fp16` in `STRATA_SET`** appends `--kv fp16` to the engine argv after setup's own
  `--kv int8` is stripped, so `desc_matches()` keeps matching on every start. Verified in this image
  2026-10-25 (the engine argv carries `--kv fp16`).
- **`PARALLEL` still defaults to "fit to the card"** (3 on 24 GB) and `strata launch` turns that into
  a profile override → `--batch 3` (batch slots are otherwise opt-in). Set `PARALLEL=1` for the
  single-request lane (the swap's `concurrencyLimit: 20` is only its queue depth).
- **Session files are unavailable on this split lane, for a different reason than in 0.2.0.**
  `--slot-save-path` is a `strata serve` option, not an engine flag; passing it through `STRATA_SET`
  sends it to the engine and fails with `unknown argument`. Engine 0.1.52 does implement
  `SAVE`/`RESTORE`, but it refuses session files with a layer split
  (`session files do not support --layer-split`; also `--batch`, `--peer-device`). Our production
  pair is a split, so only parking (`--conversation-cache-*`) works. Do not pass the flag.
- No `--memory` cap: 0.2.7 still plans from the host's RAM and the entrypoint warns when a cgroup cap
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
  ghcr.io/hurui200320/llm-ops-strata:0.2.7-rocm-gfx110x sh
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

## Building the 0.2.7 image (with the RX 7900 DeepSeek engine)

Run **Actions → strata-image → Run workflow**. [`.github/workflows/strata-image.yml`](../../.github/workflows/strata-image.yml)
is the source of truth: its header says why the image exists and what it changes against an upstream one, its `env:`
block pins the Strata tag, the engine release, the TheRock ROCm line and the hipBLASLt table, and its steps are the
build and its gates (signed fetch, package checks, the `strata-dsv41` build, the in-image smoke tests). The two
remaining switches: `include_dsv41` (add the RX 7900 DeepSeek binary) and `push` (off builds and smoke-tests without
publishing). The run summary carries the digests.

The image fedora-tuf runs is the 0.2.7 build; its run number, digest and package hashes are in
[notes/strata.md](../../notes/strata.md#pins). Earlier pins: 0.2.6 (`0.2.6-rocm-gfx110x`, run
40313031068), 0.2.5 (`0.2.5-rocm-gfx110x`, run 40188533610), and 0.2.4/0.2.4.2
(`0.2.4-rocm-gfx110x`, superseded by 0.2.4.2).

A local build is those same steps run by hand (Docker + network, no GPU): upstream's `docker/fetch-engines.py`,
[build-dsv41.sh](build-dsv41.sh) and `docker/build.sh`. Tags **v0.2.4.1 through v0.2.6** needed the
nested-`tools/` context fix applied by hand (without it the Terminal stage stopped at
`installer tool missing: tools/dsv41/tune.py`); **v0.2.7 carries the fix upstream** (PR #3974), so the
workflow applies no patch and neither does a local build.

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
with the macro **`docker-server-strata-rocm`** and the models appended at the end of `models:` —
vanilla **`qwen3.8-flash-next-iq3s`**, SC117 **`qwen3.8-flash-next-sc117-iq3s`** and DeepSeek
**`deepseek-v4.1-flash-2bpw`**. `${PORT}` is assigned in **model-ID order** (llama-swap sorts the
IDs, not the file): the DeepSeek lane is 10001 and the two Qwen lanes are 10008/10009; those
per-lane ports are internal to the proxy (clients use `:8080`). `CHAT_TEMPLATE_PATH` and the picture
budget `VISION_TOKENS`/`VISION_MIN_TOKENS` are per model entry, not in the macro (each lane names its
own template and budget). Implementation record and current pins:
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
- **Keyless, on a Docker bridge.** Strata 0.2.7 still makes the API key optional; upstream's entrypoint
  still hardcodes `--access lan` (0.0.0.0). This lane runs on the default Docker bridge with
  [entrypoint-local.sh](entrypoint-local.sh), a fork of upstream's entrypoint that passes
  `--no-api-key` when no key is set, suppresses upstream's false "Strata made one" warning,
  installs `CHAT_TEMPLATE_PATH` over every pack's chat template, and writes the picture-reader
  token budget (`VISION_TOKENS`/`VISION_MIN_TOKENS`) into the run settings. Docker
  publishes only `127.0.0.1:${PORT}` (`-p 127.0.0.1:${PORT}:${PORT}`) on the host, where llama-swap
  proxies; there is no `--network host`. `launch` stores no key, so llama-swap stays `apiKeys: []`
  and clients send no header; the LAN reaches Strata only through llama-swap. **The macro passes
  `-e STRATA_ALLOWED_HOSTS=fedora-tuf`** because 0.2.7 adds DNS-rebinding Host protection for
  keyless servers: llama-swap forwards the client's Host header, and without the flag
  `http://fedora-tuf:8080/...` (and the `/upstream` UI) is answered with
  `403 Host ... is not allowed` while a LAN-IP URL still works. The browser UI works
  at `http://<fedora-tuf>:8080/upstream/qwen3.8-flash-next-iq3s/` (the app uses relative asset and
  API paths, which llama-swap's prefix stripping preserves).
- **Re-diff the entrypoint fork on every image bump.** Extract the new
  `/opt/strata/entrypoint.sh` from the image
  (`docker run --rm --entrypoint /bin/sh IMAGE -c 'cat /opt/strata/entrypoint.sh'`) and compare it
  with `entrypoint-local.sh`; re-apply the six changes (the `--no-api-key` conditional, the final
  keyless warning, the pack chat-template swap, the `STRATA_DSV41_EXE` export for the DeepSeek
  engine, and the picture-reader token budget) and nothing else. 0.2.7's `docker/entrypoint.sh` adds only
  `legacy_models_guard` (#3421) to 0.2.6's, so the diff is expected to be exactly those six changes plus that
  upstream addition (the fork carries it unchanged). The fork must exist in the fedora-tuf checkout before the first start
  (mode 755): a missing bind source is created by Docker as a root-owned directory and the
  container dies on `--entrypoint`.
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
  (`907,542,944 B`, sha256 `2e788f8c…`), not the ISTA pin. 0.2.7 changes the engine folder (0.1.50 -> 0.1.52),
  so the lane drops its record and re-runs setup once; it reuses the pack and the lane's own mmproj — no
  repack and no download (the 0.2.5 pattern). Vision with this copy was verified with
  a 640×360 image after the 0.2.2 deployment; the 0.2.3 smoke repeated the check on the vanilla
  lane. Despite the unchanged GSQ-RCO layout the ISTA pack cannot be reused (95 tensors changed
  type, 1,214/1,223 offsets moved). Details: `notes/strata.md`.
  Parking knobs already ride in `STRATA_SET`; current per-lane settings are in `notes/strata.md`.

## DeepSeek (dsv41) lane

**`deepseek-v4.1-flash-2bpw`** (sorts first by model ID, so `${PORT}` = 10001; the two Qwen lanes are
10008/10009) serves the IST-DASLab 2.00 bpw set of DeepSeek-V4.1-Flash on the image's `strata-dsv41`
engine. Same container shape as the Qwen lanes (entrypoint fork, keyless bridge, `-t 120`,
`/health`), with these differences:

- **Engine discovery.** In this image the Terminal never finds the added binary on its own (setup
  looks in `engine-dsv41/` stores, and in a container `ROOT/engine` does not exist). Entrypoint fork
  change 4 exports `STRATA_DSV41_EXE=/opt/strata/components/engine/<package>/strata-dsv41` when it
  is unset; setup's `dsv41_exe()` and the launch's `dsv41_engine_found()` both honour it. Without
  it, first setup stops at "DeepSeek's engine is not available for this computer yet".
- **RAM budget and VRAM reserve.** `STRATA_SET="ssd-cache=36 reserve-mib=896"` → a
  36 GB (decimal) RAM store for the hot CPU-lane experts, and 896 MiB per GPU kept out of the expert
  cache at load (unset, the store auto-sizes to available RAM − 12 GB, capped at 48 GB).
  `reserve-mib` is a plan-time input, not a free-VRAM floor. Why these values, and what was measured:
  [notes/deepseek/v41f1017.md](../../notes/deepseek/v41f1017.md). The Qwen lanes'
  `resident-budget-gib` is a Flash-Next setting that DeepSeek ignores (setup warns about it);
  `--ssd-cache` is the knob.
- **Only dsv41 settings work here.** `STRATA_SET` lowercase keys become engine argv flags and
  `strata-dsv41` exits 2 on an unknown one, so the Qwen keys (`kv=`, `vram-reserve-mib=`,
  `vram-reserve-later-mib=`, `conversation-cache-mib/slots=`, `resident-budget-gib=`,
  `prompt-cache-spread=`, `pipeline-windows=`, `layer-split=`, `trim-stage-weights=`,
  `spec-min-p=`) crash-loop the lane; `layer-split=` is even parsed and ignored. A manual split is
  `DS4_SPLIT=a,b` (env). `dspark=on` / `threads=on` are stripped to a bare flag by the launch and eat
  the next option — use `dspark=1` / `threads=1`. Since 0.1.50 `--experts N` (5 = Faster, 6 = Full
  quality) also exists; this lane keeps the default 6 (`--speed`/`--full-quality` are launch flags,
  not STRATA_SET keys).
- **gfx11 WMMA prompt kernels: `STRATA_DS4_HIPK=1`.** 0.1.50 enables them by default on gfx1151
  only; on gfx1100 they are opt-in and were enabled after the measured pp gain at ~51K tokens (same
  answers, decode within noise). Pass it as a **container env var** (the engine reads it directly;
  `=0` turns the kernels off everywhere). 0.1.52 keeps this split (gfx12 default, gfx1151 default,
  gfx1100 opt-in) and adds the **RDNA3 WMMA dense GEMM** before hipBLAS (`STRATA_DENSE_PFGEMM`),
  on by default on gfx11 — a new prompt path on this card. Full A/Bs:
  [notes/deepseek/v41f1017.md](../../notes/deepseek/v41f1017.md).
- **Speed.** The measured pp/tg curve, the tuning arms and the rejects (manual splits, chunk 16384,
  forced DSpark), the 0.2.5 engine behaviour, the 0.1.51 copy-all rule (not reachable on this host),
  the 0.1.52 prompt stack (no post-prompt store settle, staging ring, auto attention slab, RDNA3
  dense GEMM) and the `STRATA_DS4_HIPK=1` A/B:
  [notes/deepseek/v41f1017.md](../../notes/deepseek/v41f1017.md). `DS4_DEFAULTS=026` (container env)
  restores every 0.2.6 default in one variable; `DS4_SETTLE_STORE=1` brings back the post-prompt
  store pre-fill (its absence makes the first words after a long prompt slower).
- **Vision (0.2.7).** The engine has its own tower (319 MiB of code, loaded with the model; no
  separate reader process). Setup packs the tower's weights from `mmproj-BF16.gguf` (972,208,288 B,
  sha256 `9a1ef7c4…` — the 0.2.7 pin, and the file already in the lane's model folder), so the setup
  pass repacks the pack with `vision.*`/`image_token_id` (the pack stays ~8.5 MB; the tensors are
  read from the mmproj in place). **Verified on this pair 2026-10-28** (first gfx1100 run of the
  tower; upstream checked one R9700 only): red/blue-halves picture answered
  `left=red, right=blue`, first 14.0–14.4 s, warm 1.6 s, text after fine, and the lane advertises
  `capabilities.in: [text, image]`. Runtime cost and the A/B: see the dsv41 note. `--vision no` at
  setup leaves it out. **Picture budget: `DS4_VISION_TOKENS=1140`** (fork change 6) — one prompt
  token per 42×42 px cell, span `n_h*(n_w+1)+2`; setup's native 1024 downscales a 4K screenshot to
  1680×966, 1140 is the tower's hard maximum (10,240-patch attention; 1378×1378 square / 1840×1008
  at 16:9) and costs only ~17 MB more tower memory. The knob writes the pack's
  `vision.max_image_tokens` at every start; the Qwen `VISION_TOKENS` only patches Flash-Next readers.
  Token math, the resolution table and why the Qwen 8192-token budget cannot be matched (DeepSeek's
  tower caps at 1,140): [notes/strata.md](../../notes/strata.md) ("Context, KV, parallel, vision").
- **No KV or parallel.** The engine has its own cache format (no `--kv` flag at all); `--parallel` is
  ignored with a warning in this version. Do not pass `KV=` or `PARALLEL`; the lane declares
  `capabilities` explicitly.
- **Context 256k.** `CONTEXT=256k` → `--max-context 262144`; upstream checked 4K/8K only, so setup
  warns and keeps it as asked. Fallback ladder: 128k, then 64k (the pack, table and drafter are
  reused; only the install record is remade).
- **First start** (`healthCheckTimeout: 1200`): setup runs outside llama-swap — a one-time SHA-256
  pass over the files (~257 GB), a 5.8 MB Unsloth metadata-shard download, the pack + hot-expert
  profile, and the 16-token smoke (the first gfx1100 run of the binary). 0.2.7 changes the engine
  folder (0.1.50 -> 0.1.52), so the lane drops its record and runs setup once; the pack is rebuilt
  with the vision tower (the weights are the lane's own `mmproj-BF16.gguf`, no download). **Pass
  `-e VISION=on` on that setup run**: a lane installed under 0.2.6 saves `"vision": "off"` in its
  preset and setup only packs the tower when pictures are on (without the flag the pass silently
  reuses the text-only pack). Also give the init container the GPUs to itself — a loaded Qwen lane
  makes the smoke fail with `cudaMalloc 2 MiB on CUDA0 failed`. Later starts take a few minutes.
  Full record: [notes/strata.md](../../notes/strata.md); speed:
  [notes/deepseek/v41f1017.md](../../notes/deepseek/v41f1017.md).

  The init run (stop the container when setup finishes; the lane reuses the prepared tree):

  ```sh
  docker run -d --name strata-init-ds --init --rm \
    -p 127.0.0.1:10109:8080 \
    --entrypoint /entry-local.sh \
    --device /dev/kfd --device /dev/dri \
    --group-add "$(getent group video | cut -d: -f3)" \
    --group-add "$(getent group render | cut -d: -f3)" \
    --security-opt label=disable --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
    -e PORT=8080 -e VISION=on \
    -e STRATA_HIPBLASLT_TABLE=on \
    -e CHAT_TEMPLATE_PATH=/templates/deepseek/v4.1-flash-1017-chat-template.jinja \
    -v /mnt/unenc-xfs/llm-ops/deploy/strata/entrypoint-local.sh:/entry-local.sh:ro \
    -v /mnt/unenc-xfs/llm-ops/templates:/templates:ro \
    -v /mnt/unenc-xfs/strata/ISTA-DASLab-dsv41f1017:/data \
    -v /mnt/unenc-xfs/models:/models:z \
    -e MODEL=IST-DAS-2.00bpw -e FAMILY=deepseek41 -e CONTEXT=256k -e GPUS=0,1 \
    -e MODEL_PATH=/models/strata/ISTA-DASLab-dsv41f1017 \
    -e STRATA_SET="ssd-cache=36 reserve-mib=896" \
    -e STRATA_DS4_HIPK=1 \
    -e DS4_VISION_TOKENS=1140 \
    ghcr.io/hurui200320/llm-ops-strata:0.2.7-rocm-gfx110x
  docker logs -f strata-init-ds
  ```

  The host port is 10109 (8080 belongs to llama-swap, and 10001–10009 are the lanes' internal
  ports); inside the container the server uses `PORT=8080` like the other manual runs.
  `--entrypoint /entry-local.sh` is the fork, which also resolves `STRATA_DSV41_EXE` — a manual run
  through the upstream entrypoint would stop at the engine check.

## Chat template override

Strata has no `--chat-template-file` (unlike `llama-server`). Setup extracts the chat template into
the prepared pack at `<data>/Strata-data/packs/<pack>/tokenizer/chat_template.jinja`, and the server
reads that copy at every start. A one-time swap survives re-installs, but a fresh pack or a
pack-format bump re-extracts the GGUF's own template — so every lane re-applies its own repo copy
on every start:

- `entrypoint-local.sh` compares `CHAT_TEMPLATE_PATH` with every pack's copy and replaces it when it
  differs (silent when equal; warns when no pack exists yet).
- The macro mounts the repo's `templates/` at `/templates:ro` — a directory mount, like the llama.cpp
  lanes: a single-file mount would go stale because `fetch-templates.sh` replaces the file by rename —
  and each model entry names its own `CHAT_TEMPLATE_PATH` (the Qwen lanes
  `/templates/qwen3.8/Flash-Next-chat-template.jinja`, the DeepSeek lane
  `/templates/deepseek/v4.1-flash-1017-chat-template.jinja`).

Updating the template: run `templates/fetch-templates.sh`, commit, `git pull` in
`/mnt/unenc-xfs/llm-ops`, then restart the lane. Nothing is repacked and no GGUF is re-downloaded. A
start that runs setup (fresh pack, repack) serves the GGUF's template once (the fork warns only when
no pack exists yet); the next start installs the repo copy. After a template change on a Qwen lane,
verify a short chat, a declared tool call and the `:npt` alias (its
`chat_template_kwargs.preserve_thinking` must still be honoured); on the DeepSeek lane, verify a
short chat and a tool call through llama-swap. Record the result in
[notes/strata.md](../../notes/strata.md).

## Vision token budget

Setup writes the picture reader's budget into the lane's run settings (`/data/install/strata-<tag>.json`,
the `"vision"` section: `max_tokens`, and `min_tokens` as a key setup does not write) and the server
passes it to the reader as `--max-tokens` / `--min-tokens` at every start. Setup's defaults are 1024
(GPU reader) and 300 (CPU reader); there is no per-launch option (`strata launch --set` reaches the
engine only). Each Qwen model entry therefore sets `VISION_TOKENS` / `VISION_MIN_TOKENS`, and
entrypoint fork change 5 writes them into every run config that has a `"vision"` section before the
server starts:

- Both lanes run `VISION_TOKENS=8192 VISION_MIN_TOKENS=1024`: at most 8192 image tokens per picture,
  smaller ones scaled up to 1024. The reader sizes a picture into that budget at one token per
  1024 px, so 8192 ≈ an 8.4 MP picture (a 4K screenshot fits) — the same 1024-8192 the Qwen3.8 27B
  llama.cpp lanes use. Background and token math: [notes/strata.md](../../notes/strata.md).
- A config this image has to make from nothing (first install, an engine-folder change, a changed
  `GPUS`) serves setup's default once; the next start applies the budget. Setup keeps the value when
  it writes the config again (same reader device for `max_tokens`; `min_tokens` is carried over), so
  no manual edit is needed and the repo stays the source of truth.
- Verification: the start log prints `Vision budget: strata-<tag>.json: max_tokens=…, min_tokens=…`
  when it changes anything, and the reader process carries
  `--max-tokens 8192 --min-tokens 1024`. A bigger budget costs encoder VRAM and time per encode (the
  reader's warm-up runs at 4096 tokens with this cap); watch the engine's free-VRAM line after a large
  picture before raising `vram-reserve-mib`.

The DeepSeek lane has no reader: its engine's own tower sizes each picture from the pack
(`vision.max_image_tokens`, one token per 42×42 px cell, span `n_h*(n_w+1)+2`), and **fork change 6**
writes that key from `DS4_VISION_TOKENS` / `DS4_VISION_MIN_PIXELS`. The lane runs
`DS4_VISION_TOKENS=1140`, the tower's hard maximum (10,240-patch attention; 1141 refuses to load);
see "DeepSeek (dsv41) lane" and the token math/resolution table in
[notes/strata.md](../../notes/strata.md).

## The hipBLASLt tuning table (packaged since 0.2.1)

The gfx1100 table routes the dense prompt GEMMs through hipBLASLt (docs/AMD_HIP.md: +40-60% prompt
speed on the 7900 XTX); its solution ids are valid only for the hipBLASLt version the engine runs
with. Since 0.2.1 the AMD engine package carries `tuning/gfx1100-hipblaslt-100200.txt` (listed and
hashed in `engine-component.json`) and records `hipblaslt_version: 100200` in `BUILD.json`. The
server resolves the table for the card at start (`serve/server.py` with
`strata_control/hipblaslt_tuning.py`) and passes it to the engine; the engine validates
magic/arch/version itself and, since 0.2.3, also falls back to the table beside its own exe when
`STRATA_HIPBLASLT_TUNING` is absent (`STRATA_HIPBLASLT_TABLE=off` still turns it off).

The 0.1.50 engine package **re-tuned** the table for the same hipBLASLt version `100200` (the table
hash is in [notes/strata.md](../../notes/strata.md#pins)) — the version gate and the log lines are
unchanged but pp numbers can move slightly; the 2026-10-25 spot-check
([notes/qwen3.8/Flash-Next.md](../../notes/qwen3.8/Flash-Next.md)) is the reference. The 0.2.7
package is still built against hipBLASLt 1.2.0 (`docs/AMD_HIP.md` at the tag); the workflow's gate
verifies the packaged table against the manifest and prints its hash, which is recorded in
[notes/strata.md](../../notes/strata.md#pins).

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
