# Strata on fedora-tuf

Strata (upstream [`Niko1221/Strata`](https://github.com/Niko1221/Strata)) serves Qwen3.8 Flash-Next
(125B-class MoE, IQ3_S GGUF) on the 2× RX 7900 XTX pair. It runs Docker-only from
[our pinned build](../deploy/strata/README.md) of upstream's 0.2.4.2 Terminal around the
signed-feed 0.1.48 engine package (upstream has published no 0.2.4.x image), behind the host
llama-swap; the host needs only the `amdgpu` kernel driver — no ROCm, no compiler. The image adds
one file the two Qwen lanes never invoke but the DeepSeek lane runs: the gfx110x `strata-dsv41`
build (DeepSeek V4.1-Flash; never executed on gfx1100 hardware before this deploy).

Three lanes share the pair through a llama.cpp-style layer split, each in its own container and its
own `/data` tree (the DeepSeek lane pairs its two cards with `--tp 2` instead; see its section):

| llama-swap ID | Model | Port | Model folder | Data tree |
| --- | --- | ---: | --- | --- |
| `qwen3.8-flash-next-iq3s` | vanilla ISTA-DASLab IQ3_S | 10007 | `/mnt/unenc-xfs/models/strata/ISTA-DASLab-Flash-Next` | `/mnt/unenc-xfs/strata/ISTA-DASLab-Flash-Next` |
| `qwen3.8-flash-next-sc117-iq3s` | SC117 abliterated IQ3_S | 10008 | `/mnt/unenc-xfs/models/strata/sc117-Flash-Next-uncensored` | `/mnt/unenc-xfs/strata/sc117-Flash-Next-uncensored` |
| `deepseek-v4.1-flash-2bpw` | ISTA-DASLab DeepSeek-V4.1-Flash IQ2_XXS (2.00 bpw) | 10009 | `/mnt/unenc-xfs/models/strata/ISTA-DASLab-dsv41f1017` | `/mnt/unenc-xfs/strata/ISTA-DASLab-dsv41f1017` |

The two Qwen lanes use 262144-token context, F16 KV, `PARALLEL=1`, a 28 GiB locked resident
budget, and the same layer split across the cards. All three containers are keyless on the default
Docker bridge with only `127.0.0.1:${PORT}` published; clients and the browser UI reach them
through llama-swap on `:8080`.

The two Qwen lanes serve through llama-swap (one loaded at a time) on image 0.2.4.2 (Terminal
0.2.4.2 / Control 0.1.49.2, engine 0.1.48), switched from 0.2.3 on 2026-10-24 (commit `4c3106e`);
the lane tuning is from 2026-10-22 and carried over unchanged. 0.2.4.2 spot-check (vanilla, through
llama-swap): 60 % pp/tg 2100.5/64.05, 85 % 2172.4/64.00 (0.2.3: 2077.3/65.95 and 2132.5/63.30 —
within noise). The full runbook is [`deploy/strata/README.md`](../deploy/strata/README.md); speed
and quality measurements are in [`notes/qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md). The
DeepSeek lane was added to the config on 2026-10-25; its deploy and first measurements are recorded
in "DeepSeek-V4.1-Flash lane (dsv41)" below.

## Pins

| What | Value |
| --- | --- |
| Image | `ghcr.io/hurui200320/llm-ops-strata:0.2.4.2-rocm-gfx110x`, digest `sha256:c67827ee283e5e90be31e6c01925727256a26e76aa5b6ee15f57762fa29cf3bc`, 1,889,591,459 B on fedora-tuf (`docker inspect .Size`; the dev box reports 1,466,884,951 B for the same digest — storage driver); public GHCR package (no login) |
| Terminal | tag `v0.2.4.2`, commit `dfa3f4025c9f1bc4af8cee201b20a2d223061813`; `strata 0.2.4.2 (build 0.1.49.2; python 3.12.15)` |
| Engine | `strata-engine-0.1.48-linux-x86_64-hip-rx7900` from the `v0.2.4` release's signed feed (the 0.2.3 file re-attached), 242,858,823 B, asset sha256 `13bcf38c…`, manifest `build_id 6f8800aefe27+rocm`, variant `rx7900`, 580 listed files (582 on disk = 580 + `engine-component.json` + our `strata-dsv41`); `engine-component.json` sha256 `59a2ac74…` and `BUILD.json` `7108976a…`, byte-identical to 0.2.3 |
| `strata-dsv41` | our gfx110x build from the same `v0.2.4.2` tag, `strata-dsv41 0.1.49.2 build dfa3f4025`, sha256 `53af2567…`; beside `strata` in the engine folder, not in the manifest by design; **the DeepSeek lane runs it** via `STRATA_DSV41_EXE` (entrypoint fork change 4, verified in-image 2026-10-25); never executed on gfx1100 hardware before that lane's deploy |
| hipBLASLt table | packaged `tuning/gfx1100-hipblaslt-100200.txt`, sha256 `1bb8cd2e…`, version `100200` (unchanged from 0.2.2/0.2.3, verified 2026-10-24) |

The image is our own build of upstream's Terminal (`strata-image.yml`, run 40100350366): the
released engine package plus the one `strata-dsv41` file, so every other byte is upstream's
(public GHCR package, no login). The 0.2.3 image stays pulled for rollback. The engine folder
inside the image is `/opt/strata/components/engine/0.1.48-linux-x86_64-hip-rx7900` (upstream drops
the `strata-engine-` prefix); reference it as `component:strata-engine`, never by path.

## Host and data layout

The pair is 2× RX 7900 XTX (gfx1100, 24 GB each). GPU1 sits behind the Z790 PCH at x4; Strata's
multi-GPU mode is a layer split with hand-off through pinned host RAM, so RCCL is neither possible
nor needed. 62 GiB host RAM, Gen4 NVMe, zram and a swapfile; the locked resident budget, not a
container cap, is the memory control.

```
/mnt/unenc-xfs/
  models/
    strata/
      ISTA-DASLab-Flash-Next/        vanilla IQ3_S pair + pinned mmproj, 84.53 GB
      sc117-Flash-Next-uncensored/   SC117 IQ3_S pair + own mmproj
                                     + unused llama.cpp MTP Q8, 88.91 GB
    qwen3.8/                         unrelated 27B models
  strata/
    ISTA-DASLab-Flash-Next/          vanilla data tree, 7.9 GB
      Strata-data/packs/iq3_s/       1.5 GB native pack (+ tokenizer/chat_template.jinja)
      Strata-data/mtp/rt/            6.5 GB MTP layer
      install/strata-iq3_s.json      installed run settings
      state/                         launch state; no api-key on this keyless lane
      config/strata/launch-presets/  per-lane "docker" preset
      home/, slots/                  slots/ is empty
    sc117-Flash-Next-uncensored/     SC117 data tree, 7.9 GB
      Strata-data/packs/iq3_s/       built from SC117 shard 1
      Strata-data/mtp/rt/            copy of the vanilla 6.5 GB MTP
      install/, state/, config/, home/, slots/
    ISTA-DASLab-dsv41f1017/          DeepSeek data tree (created 2026-10-25)
      Strata-data/packs/deepseek-v4.1-flash-ist-das-2.00bpw/
                                     pack + expert-profile.bin after the first setup
      Strata-data/models/deepseek-v4.1-flash-shared/
                                     the 5.8 MB Unsloth metadata shard (--engram-meta)
      install/, state/, config/, home/, slots/
  llm-ops/                           host checkout of this repo
    deploy/llama-swap.config.yaml
    deploy/strata/entrypoint-local.sh
```

- GGUFs live once, in flat per-lane folders: download, working copy and bind-mount source are the
  same files. The model folder is mounted at `/models`; each lane's data tree is mounted at
  `/data`. Both are plain host directories — bind mounts only, no Docker volumes, so image updates
  never touch them.
- Each lane needs its own `/data` tree because both models catalog as tag `iq3_s`. SC117 is not a
  drop-in for the vanilla pack: 144 transplanted tensors changed 95 tensor types to Q8_0 and moved
  1,214/1,223 offsets, so it has a pack built from its own shard 1. Its active mmproj is its own
  907,542,944 B copy (sha256 `2e788f8c…`), not the ISTA pin (`b1a82259…`). The SC117 model folder
  also carries an unused llama.cpp Q8 MTP (4.14 GB, reclaimable).
- `Strata-data/` is the entrypoint's `--data-dir`. `install/`, `state/`, `config/`, `home/` and
  `slots/` sit beside it under `/data`. `PRESET` is left unset, so each lane keeps the image
  default `docker` preset inside its own tree.

## Init / first start

The first start (or a settings change) prepares the model; every later start goes straight to
serving. Setup extracts the chat template into the pack, builds or reuses the native expert pack
and MTP layer, writes the install record and the lane's preset. Init runs outside llama-swap so a
long pack/MTP pass does not hit the proxy's health timeout.

To rebuild or re-seed a lane, run the production environment directly. This is the vanilla command;
for SC117 change only the `/data` mount and `MODEL_PATH`:

```sh
docker run -d --name strata-init --init --rm \
  -p 127.0.0.1:8080:8080 \
  --entrypoint /entry-local.sh \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt label=disable --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
  -e PORT=8080 \
  -e STRATA_HIPBLASLT_TABLE=on \
  -e CHAT_TEMPLATE_PATH=/templates/qwen3.8/Flash-Next-chat-template.jinja \
  -v /mnt/unenc-xfs/llm-ops/deploy/strata/entrypoint-local.sh:/entry-local.sh:ro \
  -v /mnt/unenc-xfs/llm-ops/templates:/templates:ro \
  -v /mnt/unenc-xfs/strata/ISTA-DASLab-Flash-Next:/data \
  -v /mnt/unenc-xfs/models:/models:z \
  -e MODEL=IQ3_S -e CONTEXT=256k -e GPUS=0,1 -e PARALLEL=1 \
  -e MODEL_PATH=/models/strata/ISTA-DASLab-Flash-Next \
  -e STRATA_SET="kv=fp16 vram-reserve-mib=1536 vram-reserve-later-mib=768 resident-budget-gib=28 prompt-cache-spread=on conversation-cache-mib=8192 conversation-cache-slots=8 pipeline-windows=2 layer-split=26 trim-stage-weights=on spec-min-p=0.7 STRATA_SCORES_THREAD=1" \
  ghcr.io/hurui200320/llm-ops-strata:0.2.4.2-rocm-gfx110x
docker logs -f strata-init
```

Stop the container when setup finishes; the swap lane then reuses the prepared tree. A first setup
(no pack yet) makes the fork warn; a repack re-extracts the template after the swap. Either way the
next start puts the repo template in place (see the chat-template bullet).

- `MODEL_PATH` is a folder of GGUFs or a single `.gguf`; `GGUF_DIR` is an alias. Nothing is
  downloaded for the model itself. The prepared pack and MTP layer are written under
  `/data/Strata-data/`; a later start reuses them.
- Install with the default int8 KV: `setup.py` accepts only `int8|q4_0|k8v4`. F16 KV is applied at
  launch by **`kv=fp16` in `STRATA_SET`**, which replaces the installed `--kv int8` in the engine
  argv. Do not pass `KV=fp16` — `strata launch` forwards it to setup, which exits 6, and
  `desc_matches()` repeats the failure on every start.
- 0.2.4.2 keeps 0.2.3's install record (same 0.1.48 engine/vision paths and settings), so its
  first start is expected to go straight to serving: no setup pass, no repack, no download.
  (0.2.3's first start dropped the 0.2.2 record, re-ran setup once and reused the existing pack
  and MTP; a warm first start took ~40 s in the 2026-10-23 smoke, ≈30 s of it the model load.
  The same smoke verified chat, a declared-tool call, `max_tokens:-1` and 640×360 vision, and a
  warm second start needed no setup pass: stop 2.4 s, exit 0, VRAM freed.) The 2026-10-24 deploy
  confirmed it: no setup pass (`Strata-data/` stayed at 50 files and `mtp/tensors/verified.json`
  was not re-stamped), both lanes green through llama-swap, `docker stop -t 120` 3.8 s with the
  VRAM freed.
- Chat template: Strata has no `--chat-template-file` flag. Setup extracts the template at pack time
  into `<pack>/tokenizer/chat_template.jinja`, the server reads it per start, and an override is a
  file swap in the pack (it survives re-installs). The two Qwen lanes therefore run the repo's
  `templates/qwen3.8/Flash-Next-chat-template.jinja`: the entrypoint fork copies it over every pack's
  copy before `strata launch` whenever `CHAT_TEMPLATE_PATH` is set (the macro mounts the repo
  `templates/` at `/templates:ro` and sets it to `/templates/qwen3.8/Flash-Next-chat-template.jinja`).
  A start where setup re-extracts the pack template (a fresh pack, or a pack-format bump) serves the
  GGUF's own template once; the next start restores the repo copy.

## llama-swap wiring

[`deploy/llama-swap.config.yaml`](../deploy/llama-swap.config.yaml) defines the shared macro
**`docker-server-strata-rocm`** and appends the three models at the end of `models:` (`${PORT}` is
assigned in config order, hence 10007/10008/10009). `CHAT_TEMPLATE_PATH` is per model entry, not in
the macro: each lane names its own template (the two Qwen lanes
`/templates/qwen3.8/Flash-Next-chat-template.jinja`, the DeepSeek lane its own). Host llama-swap runs with `-watch-config`, so entry
changes are picked up live; no environment variable is needed to load a lane.

- Container shape: `--name ${MODEL_ID} --init --rm`, `-p 127.0.0.1:${PORT}:${PORT}`,
  `--entrypoint /entry-local.sh`, `--device /dev/kfd --device /dev/dri`,
  `--group-add 39 --group-add 105` (host gids — see below), `--security-opt label=disable
  --security-opt seccomp=unconfined`, `--ulimit memlock=-1:-1 --shm-size 2g
  --stop-timeout 120`.
- Env: `PORT`, `STRATA_HIPBLASLT_TABLE=on`, `MODEL=IQ3_S`, `CONTEXT=256k`, `GPUS=0,1`,
  `PARALLEL=1`, `MODEL_PATH`, `CHAT_TEMPLATE_PATH` (the Qwen template), and the lane's
  `STRATA_SET`. `PRESET` is unset.
- Mounts: the entrypoint fork read-only at `/entry-local.sh`, `/mnt/unenc-xfs/models:/models:z`,
  the repo's `templates/` at `/templates:ro` (Strata never reads it; the fork copies the lane's
  `CHAT_TEMPLATE_PATH` out of it), and the lane's data tree at `/data`. No named volumes, no
  `--memory` cap, no `--network host`.
- Per-model knobs: `cmdStop: docker stop -t 120 ${MODEL_ID}`; `checkEndpoint: /health`;
  `healthCheckTimeout: 600` (cold start 1–3 min, about 30 s warm, and it bounds the graceful stop);
  `concurrencyLimit: 20` (llama-swap queue depth; the engine still has one slot).
  `capabilities` is declared rather than probed because Strata's `/v1/models` omits `owned_by`.
  The `:npt` alias uses `chat_template_kwargs.preserve_thinking: false`.
- Fork: `deploy/strata/entrypoint-local.sh` is upstream's entrypoint plus four changes — pass
  `--no-api-key` when no key is set, suppress upstream's false "Strata made one" warning, replace
  each pack's `tokenizer/chat_template.jinja` with `CHAT_TEMPLATE_PATH` when it is set, and export
  `STRATA_DSV41_EXE` from `/opt/strata/components/engine/*/strata-dsv41` when unset (the DeepSeek
  engine, which this image's Terminal never finds on its own; see the DeepSeek section). `launch`
  stores no key, so llama-swap keeps `apiKeys: []`; the LAN reaches Strata only through the swap,
  and `/upstream/<model>/` UIs work. Re-diff the fork on every image bump; a missing bind source
  becomes a root-owned directory and kills the container.

Current `STRATA_SET` (identical on the two Qwen lanes; the DeepSeek lane passes only
`ssd-cache=40` — see its section):

```
kv=fp16
vram-reserve-mib=1536
vram-reserve-later-mib=768
resident-budget-gib=28
prompt-cache-spread=on
conversation-cache-mib=8192
conversation-cache-slots=8
pipeline-windows=2
layer-split=26
trim-stage-weights=on
spec-min-p=0.7
STRATA_SCORES_THREAD=1
```

`layer-split=26` puts layers 0–25 on the x16 card and 26–47 on the x4 card. `pipeline-windows=2`
supersedes `STRATA_SPLIT_OVERLAP` and is incompatible with engine batch slots, which is why
`PARALLEL` stays 1. The vanilla lane is fully resident at 262144 context: file tier 0.0 GiB, lend
3591/3591, GPU caches about 31.3 GiB and a 22.4 GiB RAM copy after `trim-stage-weights`.

## Operating notes

- **Start/stop**: a cold load takes 1–3 min, a warm start about 30 s. SIGTERM drains, unloads and
  frees VRAM via `strata launch --stop`; `docker stop -t 120` is sized for that.
- **Upgrades**: pull and pin the new image, back up `install/strata-iq3_s.json` and
  `config/strata/launch-presets/docker.json`, make sure no Strata model is loaded, re-diff the
  entrypoint fork, then let llama-swap reload or start the vanilla lane first. 0.2.3→0.2.4.2
  keeps the engine bytes, pack, MTP and install record: no repack and no setup pass expected.
- **The 0.2.4.x line is Terminal/Control only (the app stays 0.2.3) and has no upstream docker
  image as of 2026-10-24.** 0.2.4/0.2.4.1/0.2.4.2 add DeepSeek V4.1-Flash on a separate
  `strata-dsv41` engine (Control 0.1.49/0.1.49.1/0.1.49.2) and change nothing on these lanes'
  serve/launch/entrypoint path: everything new is gated on `arch`/`family` `deepseek41`, and the
  Qwen engine packages are the 0.2.3 files, byte for byte. 0.2.4.1 fixes a missing-bundle bug in
  the 0.2.4 Terminal (DeepSeek setup only); 0.2.4.2 is the tag our image is built from.
  Upstream's docker workflow is manual and has not run since 0.2.3, and its gfx110x images will
  never carry a DeepSeek engine (the feed ships `strata-dsv41` for cuda13 and hip-rx9070 only),
  hence our own image — see Pins and [`deploy/strata/README.md`](../deploy/strata/README.md).
- **Vision token cap (0.2.3).** The server and the reader clamp image tokens by default when
  `vision.max_tokens` is absent (CPU reader 300, GPU reader 1024; #3690). The lane's install record
  carries the reader settings setup wrote; the 2026-10-23 smoke verified a 640×360 picture.
- **Never** pass `KV=`; use `kv=` inside `STRATA_SET`. `KV=fp16` is forwarded to setup, which
  accepts only `int8|q4_0|k8v4` and exits 6, and `desc_matches()` repeats the failure on every
  start.
- **Never** pass `--slot-save-path`: session files are refused on a layer split. Conversation
  parking (`conversation-cache-*`) is the working cache mechanism.
- **`STRATA_HIPBLASLT_TABLE` must be a container env var.** Inside `STRATA_SET` it lands in the
  preset's `env` and is silently ignored. Without it prompts run plain hipBLAS; the start log
  should say `prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)`.
- **`--group-add` takes numeric host gids.** Docker resolves names against the image's
  `/etc/group`, which has no `render` and maps `video` to 44; on fedora-tuf the host values are
  39 (video) and 105 (render).
- **`PARALLEL` unset is not one request.** The image default fits the card and passes `--batch 3`
  on 24 GB. Keep `PARALLEL=1`.
- **No memory cap.** Upstream advises none; the engine plans from host meminfo and the entrypoint
  warns on `memory.max`. The 28 GiB resident budget is the control.
- **No volumes contract.** A missing shard fails fast by name at `check_shards`/`GGUF_DIR`, but a
  missing mmproj is not a hard failure — setup would silently download about 907 MB into the data
  tree. Keep each lane's GGUFs and mmproj co-located in its flat folder. `--gguf-dir` scans one
  flat folder: exact published name first, else the `*-00001-of-*` glob.
- **Integrity**: per-file `.sha256` sidecars are bare 64-character hex digests (no filename, no
  newline). Verify with
  `printf '%s  %s\n' "$(cat f.gguf.sha256)" f.gguf | sha256sum -c -`. Setup pins no hashes for
  the ISTA files; the sidecar pass is the integrity gate.
- **Chat template**: the two Qwen lanes run the repo's `templates/qwen3.8/Flash-Next-chat-template.jinja`
  (byte-identical to HF `Qwen/Qwen3.8-Flash-Next` main on 2026-10-24, 8,952 B), not the one embedded
  in the GGUFs. Strata reads the pack's `tokenizer/chat_template.jinja` at every
  start and has no template flag; the fork swaps the file before `strata launch` (see "Init / first
  start"). To update: `templates/fetch-templates.sh`, commit, `git pull` in `/mnt/unenc-xfs/llm-ops`,
  restart the lane. The GGUF is never re-downloaded or repacked. Verified on fedora-tuf 2026-10-24:
  both packs were rewritten right after the pull, a deliberately diverged pack was replaced at the
  next start (`Chat template: replaced iq3_s/tokenizer/chat_template.jinja …` as the first log line,
  `cmp` identical afterwards), and a short chat, a declared tool call
  (`get_weather{"city":"Paris"}`) and the `:npt` alias all answered through llama-swap.
- **API key**: the swap lanes are keyless and have no `/data/state/api-key`. A manual run with
  `API_KEY` set still creates and uses a key.

## DeepSeek-V4.1-Flash lane (dsv41)

Added 2026-10-25; **the deploy and the first measurements are pending** (the lane entry and the
entrypoint fork are in the repo; the host checkout must be updated, then the init run follows). It
serves the IST-DASLab **2.00 bpw** set (IQ2_XXS) of DeepSeek-V4.1-Flash (1017B-parameter MoE, 40
layers × 384 experts, top-6) on the image's own `strata-dsv41` engine.

### Model files and pins

| File | Size (B) | sha256 (sidecar, matches upstream's pin) |
| --- | ---: | --- |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-0000{1..4}-of-00004.gguf` (shards) | 39,075,714,624 / 39,388,449,504 / 39,246,891,616 / 20,527,837,312 | `d46efb67…` / `cec1b2c3…` / `9d9c2140…` / `9b790a22…` |
| `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (the n-gram table; read from NVMe, never held in RAM) | 110,595,290,816 | `eb19f30d…` |
| `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` (DSpark drafter) | 7,976,914,240 | `6ba6ac1b…` |
| `mmproj-BF16.gguf` (packed; pictures unused in this engine version) | 972,208,288 | `9a1ef7c4…` |
| `DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf` (**not on disk**) | 5,780,589 | `cc15d470…` — `--engram-meta` only; setup downloads it once into `Strata-data/models/deepseek-v4.1-flash-shared/` |

All seven present files were checked against upstream's `setup.py` pins on 2026-10-25 (sha256 plus
the exact pinned sizes; the shards sum to the pinned 138,238,893,056 B set). 2.00 bpw is 138 GB of
model + 111 GB table + 8 GB drafter = 257 GB decimal. The table and the drafter are shared by every
IST-DAS set (one copy serves 1.75/2.00/2.25).

### Engine and the docker discovery gap

`strata-dsv41` is our gfx110x build from the same `v0.2.4.2` tag (`0.1.49.2 build dfa3f4025`,
sha256 `53af2567…`), placed beside `strata` in the engine package folder — the shape an upstream
`DSV41=1` rx9070 package has (see Pins). That shape is auto-found only by a desktop install
(`ROOT/engine` *is* the package folder there). **In this image it is not found**: the Terminal's
launch runs setup with `STRATA_CONTROL_INSTALL=1` / `STRATA_INSTALL_DIR=/data/install`, so
`dsv41_exe()` looks in `/data/install/engine-dsv41/` and `/opt/strata/terminal/engine/`, and the
launch's `dsv41_engine_found()` probes only `engine-dsv41/` stores — verified in-image 2026-10-25.
Without a fix, first setup stops at "DeepSeek's engine is not available for this computer yet"
(the signed feed ships `strata-dsv41` for cuda13 and hip-rx9070 only).

Fix: **entrypoint fork change 4** exports `STRATA_DSV41_EXE` from
`/opt/strata/components/engine/*/strata-dsv41` when it is unset (an explicit `-e` still wins; inert
for the Qwen lanes, whose configs are not `arch: deepseek41`). Verified in-image 2026-10-25: with
the fork, `strata launch ... --plan` reports `DeepSeek engine: installed, 0.1.48` and setup-side
`dsv41_exe()` returns the binary; without it the plan warns that setup would have to download the
engine. `strata-dsv41 --version` runs in the image, but the binary had **never executed a kernel on
gfx1100 hardware** before this lane — setup's 16-token smoke is that test.

### Memory plan and the RAM budget

`strata launch ... --plan --gpu 0,1` (2026-10-25, the real pair):

- 5,202 of 15,360 experts (45 GB) in VRAM, the other 10,158 (87 GB) on the CPU lane; `--tp 2`
  (two cards share each layer's matrices; upstream measured ~2× prompt speed vs a plain split on
  two cards).
- RAM class "tight" on the 63 GB host: the engine keeps the hot experts in its RAM store and reads
  the rest from the NVMe per token.
- Setup writes `--dspark auto --dspark-conf 0.65` (ISTA recipe) and the fit plan `--tp 2 --gpus 0,1`;
  the hot-expert profile is the shipped `data/dsv41/profile-2.00.bin` copied into the pack as
  `expert-profile.bin`.

**RAM budget**: the store auto-sizes to available RAM − 12 GB, capped at 48 GB (≈48 GB here), which
would leave too little for dev work on this box. The Qwen lanes' analogue (`resident-budget-gib=28`)
is a Flash-Next setting that setup warns about and DeepSeek ignores; the DeepSeek knob is
**`--ssd-cache` in decimal GB** (engine: `budget_bytes = N × 1e9`). The lane passes
`STRATA_SET="ssd-cache=40"` → a 40 GB store cap, leaving about 20 GiB of the 62 GiB plus reclaimable
page cache. It is a launch setting: change the number and restart the lane, no re-setup.
`--ssd-cache 0` turns the store off (every CPU expert read from the SSD per token — do not).

### Context, KV, parallel, vision

- **Context 262144 (256k)**: `--context 256k` → `--max-context 262144`. Upstream checked the engine
  at 4K/8K only; setup warns "longer is kept as you chose, but untested" and keeps it. Fallback
  ladder: 128k, then 64k (the engine default is 8192).
- **No KV setting exists**: `strata-dsv41` has no `--kv` flag — its cache is its own representation
  (~17 MB per 6.5K tokens, ≈0.7 GB at 256k). `kv=fp16` in `STRATA_SET` would append `--kv fp16` to
  the engine argv and kill it; `KV=`/`resident-budget-gib` are Flash-Next settings that setup
  rejects for deepseek41.
- `--parallel` and `--vision` are ignored with a warning in this engine version: one request at a
  time, text only. The lane declares `capabilities: in: [text]` and passes neither.
- Thinking is on per the template; `reasoning_effort: "none"` or
  `chat_template_kwargs.enable_thinking: false` turns it off. No `:npt` alias yet (that keyword is
  `preserve_thinking`, a Qwen-template thing).

### Init / first start

The first start does the setup work — run it outside llama-swap (same pattern as the Qwen lanes):

1. one-time SHA-256 pass over the ~257 GB of present files (several minutes on the NVMe);
2. the 5.8 MB Unsloth metadata shard download;
3. pack build (`Strata-data/packs/deepseek-v4.1-flash-ist-das-2.00bpw/`: tensor map, hash tables,
   tokenizer, the GGUF's own chat template) + `expert-profile.bin`; a few MB to ~6 GB of disk;
4. `strata-dsv41 --pack-info` (every file present), then the 16-token smoke whose answer must name
   Paris — the first kernel run on gfx1100 (allowed up to 5400 s; the cold load reads the experts).

The full init command is in [`deploy/strata/README.md`](../deploy/strata/README.md) ("DeepSeek (dsv41)
lane"): the Qwen init with the DeepSeek values — `MODEL=IST-DAS-2.00bpw`, `FAMILY=deepseek41`,
`GPUS=0,1`, `CONTEXT=256k`, no `PARALLEL`, no `KV`, no tuning `STRATA_SET`,
`CHAT_TEMPLATE_PATH=/templates/deepseek/v4.1-flash-1017-chat-template.jinja`, and
`/mnt/unenc-xfs/strata/ISTA-DASLab-dsv41f1017` at `/data`. Then stop the container and let
llama-swap start the lane. Later starts take a few minutes (the CPU lane's experts are read first),
hence the lane's 1200 s health timeout.

### Pending (next session)

- Init run, smoke, and a declared tool call through llama-swap.
- Warm/cold restart timing and RAM settling against the 40 GB store cap.
- 60 %/85 % pp/tg with `bench/speed/run_longctx.py --model deepseek-v4.1-flash-2bpw --ctx 262144`
  (first pass `--in-pcts 20,60,85 --reps 1`; the full default run is hours at these speeds).
- Optional afterwards: `TUNE=quick` (dsv41's own tuner, 5–10 min) and `ssd-cache`/`dspark-block`
  variations.
