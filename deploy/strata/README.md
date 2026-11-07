# Strata (AMD / HIP) deployment

The Strata image the lanes run is **our own build**, `ghcr.io/hurui200320/llm-ops-strata:0.2.11-rocm-gfx110x`,
made by [`.github/workflows/strata-image.yml`](../../.github/workflows/strata-image.yml). It carries upstream's
Terminal 0.2.11 (serve + the GGUF preparation tools and the `strata` CLI), upstream's signed-feed 0.1.57 Qwen engine
package, both picture readers (CPU and GPU), and the ROCm runtime for the whole RDNA 3
dGPU family (gfx1100 = RX 7900 XTX/XT, gfx1101 = RX 7800/7700 XT, gfx1102 = RX 7600) — byte for byte what
upstream's own `docker/build.sh` assembles from the same tag with its after-tag build-context fix `5d37280d`
(see "Building the image"); nothing is added. The host needs only the `amdgpu` kernel driver: no ROCm, no compiler.

We build our own image because upstream publishes no Docker images. See "Building the image" and the pins below;
historical context and upgrade records are in [notes/strata.md](../../notes/strata.md). (Until 2026-11-04 the image
also carried our gfx110x `strata-dsv41` DeepSeek engine; the DeepSeek lane moved to Project Maya that day
([notes/maya.md](../maya.md) 20261104) and the build was removed from the workflow
([notes/strata.md](../../notes/strata.md) 20261104).)

## Pins

Current image: `ghcr.io/hurui200320/llm-ops-strata:0.2.11-rocm-gfx110x` (Terminal `v0.2.11`, engine
`0.1.57` rx7900). Digests, package hashes, and verification records are in
[notes/strata.md](../../notes/strata.md); the workflow `env:` block pins the build inputs.

- **Engine folder**: `/opt/strata/components/engine/0.1.57-linux-x86_64-hip-rx7900` (upstream names
  the folder without the `strata-engine-` prefix; reference the engine as `component:strata-engine`,
  never by path).
- **Other package files**: 0.1.57 also ships the all-cards split engine (`strata-split-2/3/4-amd`,
  ~+49 MB). Their Auto rule never selects this pair: it requires gfx1201, exactly 4 cards and
  context ≤ 128K (0.2.11 additionally includes the IQ3_S i-quant set there, because the split now
  beats the layer split on 4× R9700 — on 2–3 cards upstream still measures the layer split faster),
  so the lanes keep the layer split. Exact build ids and hashes are in
  [notes/strata.md](../../notes/strata.md).
- **hipBLASLt table**: packaged at `tuning/gfx1100-hipblaslt-100200.txt` (TheRock `7.10.0a20251120` /
  hipBLASLt 1.2.0). On gfx1100 the table is **opt-in**: pass `STRATA_HIPBLASLT_TABLE=on` as a
  **container environment variable** (`-e`), **not** inside `STRATA_SET` (see "The hipBLASLt tuning table").

Upstream verifies every package against the signed update feed (Ed25519) and re-hashes every file
inside the build; the workflow adds fetch/manifest gates and in-image
Terminal/component smoke checks.

## Run

The first start (or a settings/engine-folder change) prepares the model; every later start goes straight
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
  -e DRAFTING=mtp \
  -e STRATA_SET="kv=fp16 vram-reserve-mib=1536 vram-reserve-later-mib=1024 resident-budget-gib=28 prompt-cache-spread=on conversation-cache-mib=8192 conversation-cache-slots=8 pipeline-windows=2 layer-split=26 trim-stage-weights=on spec-min-p=0.7 STRATA_SCORES_THREAD=1" \
  ghcr.io/hurui200320/llm-ops-strata:0.2.11-rocm-gfx110x
docker logs -f strata-init    # wait for one-time setup to finish, then stop the container
```

This uses the same keyless entrypoint as the swap lanes, so an init run does not create an API key.
Stop the container when setup finishes; the swap lane then reuses the prepared tree. A first setup
writes the GGUF's own template (the fork warns when no pack exists yet), and a repack re-extracts it
after the pre-launch swap; either way the next start installs the repo template (see "Chat template
override").

- `MODEL_PATH` takes a folder of GGUF files or one `.gguf` (`GGUF_DIR` still works as an alias).
  Nothing is downloaded for the model itself; the prepared pack and MTP layer are written under
  `/data` (`Strata-data/`). Make sure each lane's GGUFs and `mmproj` stay co-located in its flat
  folder (a missing shard fails fast, whereas a missing `mmproj` causes setup to download ~907 MB into
  the data tree).
- **Use `kv=fp16` in `STRATA_SET`, never `KV=fp16`.** Passing `KV=fp16` re-runs setup and rewrites the
  install record. The lanes stay installed with default `int8`, and **`kv=fp16` in `STRATA_SET`**
  appends `--kv fp16` to the engine argv after setup's own `--kv int8` is stripped, so
  `desc_matches()` keeps matching on every start.
- **`PARALLEL` defaults to "fit to the card"** (3 on 24 GB) and `strata launch` turns that into a
  profile override → `--batch 3` (batch slots are otherwise opt-in and disable `--pipeline-windows 2`).
  Set `PARALLEL=1` for the single-request lane (llama-swap's `concurrencyLimit: 20` is only its queue
  depth).
- **Session files are unavailable on this split lane.** `--slot-save-path` is a `strata serve` option,
  not an engine flag; passing it through `STRATA_SET` sends it to the engine and fails with
  `unknown argument`. Furthermore, the engine refuses session files with a layer split
  (`session files do not support --layer-split`; also `--batch`, `--peer-device`). Our production
  pair is a split, so only conversation parking (`--conversation-cache-*`) works. Do not pass the flag.
- **No `--memory` cap**: Strata plans from the host's RAM and the entrypoint warns when a cgroup cap
  is seen; the 28 GiB resident budget is the control.
- `--stop-timeout 120`: SIGTERM drains, unloads and frees the VRAM (`strata launch --stop`). Docker's
  default 10 s is too short for a big model.
- **Groups are gids, not names**: Docker resolves `--group-add` names against the image's
  `/etc/group` — it has no `render` entry (`--group-add render` fails) and `video` is Ubuntu's 44,
  not the host's — so use the host numbers (`getent group video render`; 39/105 on fedora-tuf).
- **API key**: with the upstream entrypoint, a manual `docker run` uses the LAN bind (`--access lan`)
  and, without `API_KEY` or `--no-api-key`, makes a key on the first start and keeps it at
  `/data/state/api-key`; a client outside the container must send it (`API_KEY=` /
  `API_KEY_FILE=` override). The llama-swap lanes and the init command above use the fork instead:
  it adds `--no-api-key`, writes no key, the container runs on a bridge network, and only
  `127.0.0.1:${PORT}` is published on the host (see "llama-swap wiring").
- **Host layout**: each lane has its own data tree mounted at `/data`:
  - `/mnt/unenc-xfs/strata/ISTA-DASLab-Flash-Next/` (Qwen3.8 Flash-Next vanilla)
  - `/mnt/unenc-xfs/strata/sc117-Flash-Next-uncensored/` (Qwen3.8 Flash-Next SC117)
  Each has `Strata-data/` (packs, MTP/shared metadata), `install/` (run settings, logs), `state/`
  (presets, launch state), `config/`, `home/`. The GGUFs stay under `/mnt/unenc-xfs/models/strata/`.
  Bind mounts only; image updates never touch them.

### Toolbox (bash)

```sh
docker run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  -v /mnt/unenc-xfs/strata/ISTA-DASLab-Flash-Next:/data -v /mnt/unenc-xfs/models:/models:z \
  ghcr.io/hurui200320/llm-ops-strata:0.2.11-rocm-gfx110x sh
```

Any command replaces the server start (`docker run … image strata launch --help`, `… image sh`).
Inside: the bundled Python 3.12, `serve/`, the preparation tools the terminal imports — the
`packaging/package_terminal.py` list (`tools/iq_pack.py`, `tools/mtp_fetch.py`, `tools/mtp_pack.py`,
`tools/mtp_rt.py`, `tools/pack_index.py`, `tools/pack_layer.py`, `tools/gguf_reader.py`,
`tools/gguf_writer.py`, `tools/strata_pack.py`, `tools/strata_tokenizer.py`, `tools/strata_mcp.py`,
`tools/canonical_xcheck.py`, `tools/draft_vocab.py`, `tools/_paths.py`) — so engine-side dev tools
such as `quant_quality.py`/`strata_inspect.py` are **not** in the image. Also: llama.cpp's gguf-py
seed (`/opt/strata/seed/third_party/llama.cpp`), `docs/TERMINAL_LAUNCH.md`, the frozen
`bench/prompts/standard-v1` fixtures, and the `strata` CLI (`strata components engine`,
`strata models list`, `strata launch --help/--plan`, `strata bench run`). `mtp_pack.py`/`mtp_rt.py`
called by hand need `STRATA_GGUF_PY=/opt/strata/seed/third_party/llama.cpp/gguf-py`. There is no
compiler in the image: engine updates are new packages, or a local package built on the host with
`packaging/build_hip_linux.sh` and passed to `docker/build.sh --engine`.
`docker exec strata-init strata status` works from outside.

## Building the image

Run **Actions → strata-image → Run workflow**. [`.github/workflows/strata-image.yml`](../../.github/workflows/strata-image.yml)
is the source of truth: its header says why the image exists and what it changes against an upstream one (one
build-context patch, see below), its `env:`
block pins the Strata tag, the engine release and the hipBLASLt table, and its steps are the
build and its gates (signed fetch, package checks, the in-image smoke tests). The one switch:
`push` (off builds and smoke-tests without publishing). The run summary carries the digests, which are recorded in
[notes/strata.md](../../notes/strata.md).

**The v0.2.11 build-context patch.** v0.2.11's `packaging/package_terminal.py` copies the Mac repack scripts
(`metal/tools/metal_repack.py`, `metal/tools/stream_experts.py`) into the Terminal bundle, but the tag's
`docker/build.sh` does not stage `metal/` into the build context, so the terminal stage dies with
`No such file or directory: '/src/metal/tools/metal_repack.py'` (our CI run `42175367343`). The workflow applies
upstream's after-tag fix `5d37280d` from
[`patches/0.2.11-metal-tools-context.patch`](patches/0.2.11-metal-tools-context.patch); it changes only what
`docker/build.sh` copies, not the Terminal's content. Drop the patch once a release contains that commit.

A local build is those same steps run by hand (Docker + network, no GPU): apply the patch to the checkout, then run
upstream's `docker/fetch-engines.py` and `docker/build.sh`.

A plain upstream image (what the workflow now produces too) builds with:

```sh
git -C <strata-checkout> worktree add /tmp/strata-v0.2.11 v0.2.11
git -C /tmp/strata-v0.2.11 apply <this-repo>/deploy/strata/patches/0.2.11-metal-tools-context.patch
tar -xzf strata-engine-0.1.57-linux-x86_64-hip-rx7900.tar.gz -C <empty-dir>
/tmp/strata-v0.2.11/docker/build.sh --flavor hip \
  --engine <empty-dir> --repo /tmp/strata-v0.2.11 \
  --version 0.2.11 --tag strata:0.2.11-rocm-gfx110x-local
```

Needs network (base image + the terminal's pinned CPython 3.12 and hash-locked wheels). `--ctx-only` keeps the
assembled context; `STRATA_DOCKER_BUILD_OPTS` adds docker-build options.

## llama-swap wiring

The host llama-swap ([deploy/llama-swap.config.yaml](../llama-swap.config.yaml)) starts this image
with the macro **`docker-server-strata-rocm`** and the models appended at the end of `models:` —
vanilla **`qwen3.8-flash-next-iq3s`** and SC117 **`qwen3.8-flash-next-sc117-iq3s`**. `${PORT}` is
assigned in **model-ID order** (llama-swap sorts the IDs, not the file): the maya DeepSeek lane
(`dsv41f1017-iq2xxs`, [maya/README.md](../maya/README.md)) sorts first and keeps port 10001,
and the two Qwen lanes are 10008/10009; those
per-lane ports are internal to the proxy (clients use `:8080`). `CHAT_TEMPLATE_PATH` and the picture
budget `VISION_TOKENS`/`VISION_MIN_TOKENS` are per model entry, not in the
macro.

- **No memory cap.** Upstream advises none and the entrypoint warns on `memory.max`; the engine
  plans from the host's `/proc/meminfo` and the 28 GiB resident budget is the real control.
- **Bind mounts only**, same tree as the manual recipe (`/data`, `/models`, `/templates`). No named
  volumes and no comgr cache: the engine is AOT-built. `/templates` is the repo's `templates/` checkout,
  read-only: Strata never reads it directly — the entrypoint fork copies the lane's template out of it
  (see "Chat template override").
- **Gids, not names**: Docker resolves `--group-add` names against the image's `/etc/group` (no
  `render`, `video` is 44) — the macro passes the host numbers (39/105 on fedora-tuf).
- **Keyless, on a Docker bridge.** Strata makes the API key optional; upstream's entrypoint
  hardcodes `--access lan` (0.0.0.0). Each lane runs on the default Docker bridge with
  [entrypoint-local.sh](entrypoint-local.sh), a fork of upstream's entrypoint that passes
  `--no-api-key` when no key is set, suppresses upstream's false "Strata made one" warning,
  installs `CHAT_TEMPLATE_PATH` over every pack's chat template, writes the picture token budget
  (`VISION_TOKENS`/`VISION_MIN_TOKENS`, fork change 4), and passes `DRAFTING` (fork change 5) to
  `strata launch --drafting`. Docker publishes only `127.0.0.1:${PORT}`
  (`-p 127.0.0.1:${PORT}:${PORT}`) on the host, where llama-swap proxies; there is no
  `--network host`. `launch` stores no key, so llama-swap stays `apiKeys: []` and clients send no
  header; the LAN reaches Strata only through llama-swap. **The macro passes
  `-e STRATA_ALLOWED_HOSTS=fedora-tuf`** for Strata's DNS-rebinding Host protection on keyless
  servers: llama-swap forwards the client's Host header, and without the flag
  `http://fedora-tuf:8080/...` (and the `/upstream` UI) is answered with
  `403 Host ... is not allowed` while a LAN-IP URL still works. The browser UI works at
  `http://<fedora-tuf>:8080/upstream/qwen3.8-flash-next-iq3s/` (the app uses relative asset and API
  paths, which llama-swap's prefix stripping preserves).
- **Re-diff the entrypoint fork on every image bump.** Extract the new
  `/opt/strata/entrypoint.sh` from the image
  (`docker run --rm --entrypoint /bin/sh IMAGE -c 'cat /opt/strata/entrypoint.sh'`) and compare it
  with `entrypoint-local.sh`; re-apply the five local changes (the `--no-api-key` conditional, the
  final keyless warning, the pack chat-template swap, the Qwen picture-reader token budget, and the
  `DRAFTING` launch flag) and nothing else. The fork must exist in the fedora-tuf checkout before the
  first start (mode 755): a missing bind source is created by Docker as a root-owned directory and the
  container dies on `--entrypoint`.
- **Preset/key state.** Each lane's `docker` preset is `"access": "lan"`, `"no_key": true`,
  `"api_key": null`; the private launch config gets `"api_key": "none"` (the server's "no key"
  sentinel), and no `/data/state/api-key` is written. A manual run with `API_KEY` still creates
  and uses a key.
- **Per-model knobs**: `healthCheckTimeout: 600` on Qwen (llama-swap bounds the graceful stop by this
  value so `cmdStop: docker stop -t 120 ${MODEL_ID}` fits), `concurrencyLimit: 20` (queue depth for
  the single engine slot so concurrent requests queue
  instead of getting HTTP 429), `checkEndpoint: /health` (answers without a key); `capabilities` is
  declared, not probed. llama-swap selects its prober from `/v1/models`' `owned_by`, which Strata
  omits, so discovery would fall back to the listing-only prober (context + text; no tools/image).
- **Env passthrough**: `MODEL`, `CONTEXT`, `GPUS=0,1`, `PARALLEL=1` (Qwen lanes), `MODEL_PATH`, and
  the `STRATA_SET` string (with `kv=fp16` on Qwen); plus the container env
  `STRATA_HIPBLASLT_TABLE=on` and `DRAFTING=mtp` (Qwen lanes). Never `KV=`, never
  `--slot-save-path`, never `DRAFTING` inside `STRATA_SET`. No entry passes `PRESET`: the
  image default `docker` is saved per lane under each tree's `config/strata/launch-presets/`.
- **Uncensored Qwen lane (`qwen3.8-flash-next-sc117-iq3s`)**: serves the SC117 abliterated IQ3_S
  transplant with the same configuration as the vanilla lane, differing only in its `/data` mount
  (`/mnt/unenc-xfs/strata/sc117-Flash-Next-uncensored`) and `MODEL_PATH`. It requires its own
  `/data` tree, native pack, and `mmproj` because 95 tensors changed type and 1,214/1,223 offsets
  moved relative to the vanilla ISTA GGUF (see [notes/strata.md](../../notes/strata.md)).

## The DeepSeek lane (moved to Maya)

DeepSeek-V4.1-Flash used to run here on our gfx110x `strata-dsv41` build (`deepseek-v4.1-flash-2bpw`,
port 10001). The lane moved to Project Maya on 2026-11-04 — faster pp and tg at the same knobs, and no
engine to compile ([notes/maya.md](../maya.md) 20261104); the strata record stays as history in
[notes/strata.md](../../notes/strata.md) and [notes/deepseek/v41f1017.md](../../notes/deepseek/v41f1017.md).
Nothing on this image serves DeepSeek any more, and the workflow no longer builds the engine.

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
  and each Qwen model entry names its own `CHAT_TEMPLATE_PATH`
  (`/templates/qwen3.8/Flash-Next-chat-template.jinja`).

Updating the template: run `templates/fetch-templates.sh`, commit, `git pull` in
`/mnt/unenc-xfs/llm-ops`, then restart the lane. Nothing is repacked and no GGUF is re-downloaded. A
start that runs setup (fresh pack, repack) serves the GGUF's template once (the fork warns only when
no pack exists yet); the next start installs the repo copy. After a template change on a Qwen lane,
verify a short chat, a declared tool call and the `:npt` alias (its
`chat_template_kwargs.preserve_thinking` must still be honoured).

## Vision token budget

For the Qwen lanes, setup writes the picture reader's budget into the lane's run settings
(`/data/install/strata-<tag>.json`, the `"vision"` section: `max_tokens`, and `min_tokens` as a key
setup does not write) and the server passes it to the reader as `--max-tokens` / `--min-tokens` at
every start. Setup's defaults are 1024 (GPU reader) and 300 (CPU reader); there is no per-launch
option (`strata launch --set` reaches the engine only). Each Qwen model entry therefore sets
`VISION_TOKENS` / `VISION_MIN_TOKENS`, and entrypoint fork change 4 writes them into every run
config that has a `"vision"` section before the server starts:

- Both Qwen lanes run `VISION_TOKENS=8192 VISION_MIN_TOKENS=1024`: at most 8192 image tokens per
  picture, smaller ones scaled up to 1024. The reader sizes a picture into that budget at one token
  per 1024 px, so 8192 ≈ an 8.4 MP picture (a 4K screenshot fits). Background and token math:
  [notes/strata.md](../../notes/strata.md).
- A config this image has to make from nothing (first install, an engine-folder change, a changed
  `GPUS`) serves setup's default once; the next start applies the budget. Setup keeps the value when
  it writes the config again (same reader device for `max_tokens`; `min_tokens` is carried over), so
  no manual edit is needed and the repo stays the source of truth.
- Verification: the start log prints `Vision budget: strata-<tag>.json: max_tokens=…, min_tokens=…`
  when it changes anything, and the reader process carries
  `--max-tokens 8192 --min-tokens 1024`. A bigger budget costs encoder VRAM and time per encode (the
  reader's warm-up runs at 4096 tokens with this cap); watch the engine's free-VRAM line after a large
  picture before raising `vram-reserve-mib`.

(The retired DeepSeek lane had no separate reader: its engine's own tower sized each picture from the
pack's `vision.max_image_tokens` — fork change 6 wrote 1140 there, the tower's hard maximum. The maya
lane mirrors that budget now; see [notes/maya.md](../maya.md) 20261104.)

## Drafting (Qwen lanes)

Both Qwen lanes run the model's own **MTP** draft layer (`DRAFTING=mtp`, entrypoint fork change 5;
`pipeline-windows=2`, `spec-min-p=0.7` in `STRATA_SET`). Upstream makes **DFlash2** the Auto
drafter for Qwen3.8-Flash-Next on gfx1100; it cannot run beside `--pipeline-windows` (the MTP's
speculative next window, which DFlash2 has no counterpart for), and that flag is our big
prompt-speed win, so **DFlash2 was tried and reverted the same day**:

- Measured on the vanilla lane, engine 0.1.56, 2026-11-03 (`run_longctx.py --ctx 262144 --reps 2`):
  DFlash2 60% pp 2,100.5 / tg 54.80, 85% pp 2,170.9 / 52.65 — against MTP 1,993.1 / 65.00 and
  2,119.2 / 64.30 on the same engine. Decode is 16–18% slower (draft acceptance 0.42–0.44 vs
  0.65), prompt within noise. The DFlash2 pack (~860 MB) stays on both data trees.
- 0.2.11 changes the rule around request slots only: a DFlash2 chosen by name now serves several
  requests at once (a drafter per request, `--draft dflash2 --batch N`); `--pipeline-windows` still
  blocks it, so the arm stays the MTP.
- The arm is **pinned**, not left to Auto: `--save-preset` stores `drafting` in the lane's preset,
  and 0.2.11's launch precedence makes the `STRATA_SET` keys explicit launch values (explicit, then
  persisted overrides, then calibrated, then Auto) — a key left out of a later launch is gone, so
  the lanes pass the full `STRATA_SET` at every start. The 0.2.10 migration that predates that rule
  wrote preset backups (`docker.json.bak-0.2.10-*`).
- **Re-test** if a later release lets DFlash2 run beside `pipeline-windows` or lifts its decode:
  set `DRAFTING=dflash2` and drop `pipeline-windows=2 spec-min-p=0.7` from the lane's `STRATA_SET`.
  Speed records:
  [notes/qwen3.8/Flash-Next.md](../../notes/qwen3.8/Flash-Next.md) and
  [notes/strata.md](../../notes/strata.md).

## The hipBLASLt tuning table

The gfx1100 table routes the dense prompt GEMMs through hipBLASLt (`docs/AMD_HIP.md`: +40–60% prompt
speed on the 7900 XTX); its solution ids are valid only for the hipBLASLt version the engine runs
with. The AMD engine package carries `tuning/gfx1100-hipblaslt-100200.txt` (listed and
hashed in `engine-component.json`) and records `hipblaslt_version: 100200` in `BUILD.json`. The
server resolves the table for the card at start (`serve/server.py` with
`strata_control/hipblaslt_tuning.py`) and passes it to the engine; the engine validates
magic/arch/version itself and also falls back to the table beside its own exe when
`STRATA_HIPBLASLT_TUNING` is absent (`STRATA_HIPBLASLT_TABLE=off` still turns it off). The workflow
gate verifies the packaged table against the manifest and records its hash in
[notes/strata.md](../../notes/strata.md).

- On gfx1100 the packaged table is **opt-in**: set `STRATA_HIPBLASLT_TABLE=on`. Without it the start
  logs `... is in this package and not used by default (faster prompts, but answers can differ in
  the last digits)` and the prompt GEMMs run plain hipBLAS.
  **The switch is read from the server process's environment** (`strata_control/hipblaslt_tuning.py`
  `mode()`: `os.environ` first, then a top-level `"hipblaslt_table"` key of the run config). So it
  must be a container env var (`-e STRATA_HIPBLASLT_TABLE=on`); putting it inside `STRATA_SET`
  silently does nothing — the key is saved into the preset's `env` and only forwarded to the engine,
  which reads `STRATA_HIPBLASLT_TUNING` (a path), not the switch.
- `STRATA_HIPBLASLT_TABLE=off` never uses a table; `auto` uses it by default only on gfx1151.
- A `STRATA_HIPBLASLT_TUNING` you set yourself always wins (the same table is fine); it remains the
  fallback for a package whose table is missing. Copy one out of the image if ever needed:
  `docker run --rm IMAGE sh -c 'cat /opt/strata/components/engine/*/tuning/gfx1100-hipblaslt-100200.txt'`.
- Start log lines to check:

      hipBLASLt tuning table: gfx1100-hipblaslt-100200.txt (faster prompts; STRATA_HIPBLASLT_TABLE=off turns it off)
      prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)

  A mismatch (`version mismatch: file=100200 runtime=<N>`) means the package's `BUILD.json`/table
  pairing is wrong; pick the matching `gfx1100-hipblaslt-<N>.txt` (or none) and re-check.
