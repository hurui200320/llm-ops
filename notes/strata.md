# Strata on fedora-tuf

Strata (upstream [`Niko1221/Strata`](https://github.com/Niko1221/Strata)) serves Qwen3.8 Flash-Next
(125B-class MoE, IQ3_S GGUF) on the 2× RX 7900 XTX pair. It runs Docker-only from
[our pinned build](../deploy/strata/README.md) of upstream's 0.2.5 Terminal around the
signed-feed 0.1.50 engine package (upstream has published no 0.2.4.x/0.2.5 image), behind the host
llama-swap; the host needs only the `amdgpu` kernel driver — no ROCm, no compiler. The image adds
one file the two Qwen lanes never invoke but the DeepSeek lane runs: the gfx110x `strata-dsv41`
build (DeepSeek V4.1-Flash).

Three lanes share the pair through a llama.cpp-style layer split, each in its own container and its
own `/data` tree (the DeepSeek lane pairs its two cards with `--tp 2` instead; see its section):

| llama-swap ID | Model | Port | Model folder | Data tree |
| --- | --- | ---: | --- | --- |
| `qwen3.8-flash-next-iq3s` | vanilla ISTA-DASLab IQ3_S | 10008 | `/mnt/unenc-xfs/models/strata/ISTA-DASLab-Flash-Next` | `/mnt/unenc-xfs/strata/ISTA-DASLab-Flash-Next` |
| `qwen3.8-flash-next-sc117-iq3s` | SC117 abliterated IQ3_S | 10009 | `/mnt/unenc-xfs/models/strata/sc117-Flash-Next-uncensored` | `/mnt/unenc-xfs/strata/sc117-Flash-Next-uncensored` |
| `deepseek-v4.1-flash-2bpw` | ISTA-DASLab DeepSeek-V4.1-Flash IQ2_XXS (2.00 bpw) | 10001 | `/mnt/unenc-xfs/models/strata/ISTA-DASLab-dsv41f1017` | `/mnt/unenc-xfs/strata/ISTA-DASLab-dsv41f1017` |

The two Qwen lanes use 262144-token context, F16 KV, `PARALLEL=1`, a 28 GiB locked resident
budget, and the same layer split across the cards. All three containers are keyless on the default
Docker bridge with only `127.0.0.1:${PORT}` published; clients and the browser UI reach them
through llama-swap on `:8080`.

The two Qwen lanes serve through llama-swap (one loaded at a time) on image 0.2.5 (Terminal
0.2.5 / Control 0.1.50, engine 0.1.50), switched from 0.2.4.2 on 2026-10-25 (commit `a916d75`);
the lane tuning is from 2026-10-22 and carried over unchanged. 0.2.5 spot-check (vanilla, through
llama-swap, two samples): 60 % pp/tg 2005.8/63.55 and 1936.1/63.65, 85 % 2049.4/65.75 and
2074.8/64.30 — the first measured rep of each 60 % sample is an outlier (1739-1901 t/s) whose
request logged an ~8.5 GB OS read from the NVMe (cold page cache after the container churn); the
steady reps (2041-2133) and a direct fixed-prompt A/B (2110.5 t/s at 147K, old and new
hipBLASLt table both 2106-2111) are within noise of 0.2.4.2's 2100.5/2172.4. The full runbook is
[`deploy/strata/README.md`](../deploy/strata/README.md); speed and quality measurements are in
[`notes/qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md). The DeepSeek lane was added to the config
on 2026-10-25 and moved to engine 0.1.50 with 0.2.5, where its gfx11 WMMA prompt kernels
(`STRATA_DS4_HIPK=1`) were measured at +18.7 % pp and enabled; details in
"DeepSeek-V4.1-Flash lane (dsv41)" below and [`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md).

## Pins

| What | Value |
| --- | --- |
| Image | `ghcr.io/hurui200320/llm-ops-strata:0.2.5-rocm-gfx110x`, digest `sha256:0bf409c527a5dd9f008d9392ada2457660b3ec2162acf68df6f4822e9b4a092d`, 1,900,632,969 B on fedora-tuf (`docker inspect .Size`; the dev box reports 1,474,728,281 B for the same digest — storage driver); public GHCR package (no login) |
| Terminal | tag `v0.2.5`, commit `fa663fabc5c2c568d6035b3ae853b38e1762865a`; `strata 0.2.5 (build 0.1.50; python 3.12.15)` |
| Engine | `strata-engine-0.1.50-linux-x86_64-hip-rx7900` from the `v0.2.5` release's signed feed, 243,072,991 B, asset sha256 `84698dbf…`, manifest `build_id 6270e2d9d6bf+rocm`, variant `rx7900`, 580 listed files (582 on disk = 580 + `engine-component.json` + our `strata-dsv41`); `engine-component.json` sha256 `43746aaf…` and `BUILD.json` `4b70ff83…` |
| `strata-dsv41` | our gfx110x build from the same `v0.2.5` tag, `strata-dsv41 0.1.50 build fa663fabc`, sha256 `3b6aa77f…`; beside `strata` in the engine folder, not in the manifest by design; **the DeepSeek lane runs it** via `STRATA_DSV41_EXE` (entrypoint fork change 4, verified in-image 2026-10-25) |
| hipBLASLt table | packaged `tuning/gfx1100-hipblaslt-100200.txt`, sha256 `235a26c3…` (873 B), version `100200`; **re-tuned for 0.1.50** (0.1.48 shipped `1bb8cd2e…`, same version) |

The image is our own build of upstream's Terminal (`strata-image.yml`, run 40188533610): the
released engine package plus the one `strata-dsv41` file, so every other byte is upstream's
(public GHCR package, no login). The 0.2.4.2 image stays pulled for rollback. The engine folder
inside the image is `/opt/strata/components/engine/0.1.50-linux-x86_64-hip-rx7900` (upstream drops
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
  ghcr.io/hurui200320/llm-ops-strata:0.2.5-rocm-gfx110x
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
- 0.2.5 changes the engine folder (0.1.48→0.1.50): the 0.2.4.2 record names 0.1.48 paths, so the
  first start drops it and re-runs setup once — **no repack and no download** (verified on vanilla
  2026-10-25: `Strata-data/` stayed at 50 files, only `mtp/tensors/verified.json` was re-stamped,
  and the record was rewritten with the 0.1.50 engine/vision paths; the first request through
  llama-swap answered after 45.8 s including the setup pass, the load and 8 generated tokens).
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
assigned in **model-ID order** — llama-swap sorts the IDs, not the file — so the DeepSeek lane is
10001 and the two Qwen lanes moved to 10008/10009 when it was added; measured 2026-10-25). These
ports are internal to the proxy; clients use `:8080`. `CHAT_TEMPLATE_PATH` is per model entry, not in
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
  `STRATA_SET` (the DeepSeek lane also sets `STRATA_DS4_HIPK=1`). `PRESET` is unset.
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
`ssd-cache=36 reserve-mib=896` plus the container env `STRATA_DS4_HIPK=1` — see its section and
[`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md)):

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
  entrypoint fork, then let llama-swap reload or start the vanilla lane first. 0.2.3→0.2.4.2 kept
  the engine bytes, pack, MTP and install record (no setup pass). 0.2.4.2→0.2.5 changes the engine
  folder (0.1.48→0.1.50): each lane's record is dropped and setup re-runs once on its first start —
  **no repack and no download** (verified on vanilla 2026-10-25; `Strata-data/` stayed at 50 files,
  only `mtp/tensors/verified.json` was re-stamped, the record now names the 0.1.50 engine/vision
  paths). The fork's template swap re-applies the repo template only when a repack re-extracted it
  (silent on this upgrade).
- **0.2.5 is a real engine release (Terminal/app 0.2.5, Control and every Qwen/DeepSeek engine
  package 0.1.50) and still has no upstream docker image: since 0.2.5 upstream ships one
  engine-less installer per OS, and its docker workflow has not run since 0.2.3.** For these lanes
  the Qwen engine 0.1.48→0.1.50 changes: the install record is dropped once (engine folder path),
  the hipBLASLt table is re-tuned for the same hipBLASLt 100200, and the engine gains CPU-row
  kernels (this host has no AVX-512, so it runs the AVX-2 rows) plus a CUDA-only adaptive-async
  default (inert on our HIP split). The dsv41 engine gains `--experts N` (5 on ≤16 GiB or shared-RAM
  PCs; our 48 GiB stays 6), the length-based deterministic prompt path with a background warm-up,
  fewer launches per layer, store-hit/miss overlap, and the gfx11 WMMA expert prompt kernels
  (opt-in on gfx1100 with `STRATA_DS4_HIPK=1` — measured 2026-10-25, see the DeepSeek section).
  The feed split is unchanged (`strata-dsv41` for cuda13 and hip-rx9070 only), hence our own image
  — see Pins and [`deploy/strata/README.md`](../deploy/strata/README.md).
- **Vision token cap (unchanged in 0.2.5).** The server and the reader clamp image tokens by default when
  `vision.max_tokens` is absent (CPU reader 300, GPU reader 1024; #3690). The lane's install record
  carries the reader settings setup wrote; the 2026-10-23 smoke verified a 640×360 picture.
- **Never** pass `KV=`; use `kv=` inside `STRATA_SET`. 0.2.5's setup now also accepts `fp16`, but
  `KV=fp16` would re-run setup once and rewrite the record. `desc_matches()` compares the KV
  choice with the installed config, so the lanes stay installed with int8 and `kv=fp16` in
  `STRATA_SET` appends `--kv fp16` to the engine argv (verified 2026-10-25).
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

Deployed 2026-10-25 (init + lane running; deploy facts below, the measured speed curve is in
[`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md)). It serves the IST-DASLab **2.00 bpw** set
(IQ2_XXS) of DeepSeek-V4.1-Flash (1017B-parameter MoE, 40 layers × 384 experts, top-6) on the
image's own `strata-dsv41` engine (0.1.50 since the 0.2.5 image, 2026-10-25).

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

`strata-dsv41` is our gfx110x build from the same `v0.2.5` tag (`0.1.50 build fa663fabc`,
sha256 `3b6aa77f…`), placed beside `strata` in the engine package folder — the shape an upstream
`DSV41=1` rx9070 package has (see Pins). That shape is auto-found only by a desktop install
(`ROOT/engine` *is* the package folder there). **In this image it is not found**: the Terminal's
launch runs setup with `STRATA_CONTROL_INSTALL=1` / `STRATA_INSTALL_DIR=/data/install`, so
`dsv41_exe()` looks in `/data/install/engine-dsv41/` and `/opt/strata/terminal/engine/`, and the
launch's `dsv41_engine_found()` probes only `engine-dsv41/` stores — verified in-image 2026-10-25.
Without a fix, first setup stops at "DeepSeek's engine is not available for this computer yet"
(the signed feed ships `strata-dsv41` for cuda13 and hip-rx9070 only).

Fix: **entrypoint fork change 4** exports `STRATA_DSV41_EXE` from
`/opt/strata/components/engine/*/strata-dsv41` when it is unset (an explicit `-e` still wins; inert
for the Qwen lanes, whose configs are not `arch: deepseek41`). Verified in-image 2026-10-25 with
0.1.48 (`strata launch ... --plan` reported `DeepSeek engine: installed, 0.1.48`; without the fork
the plan warns that setup would have to download the engine). The 0.2.5 lanes resolve it the same
way: the engine starts from the 0.1.50 package path and the launch's Speed line (`Full quality`) is
gated on `dsv41_engine_version(exe) >= 0.1.50`, which it reads from the package folder.
`strata-dsv41 --version` runs in the image; the binary had **never executed a kernel on
gfx1100 hardware** before this lane's first deploy (2026-10-25) — setup's 16-token smoke was that
test, and the 0.1.50 rebuild repeats it on the 0.2.5 setup pass.

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

**RAM budget and VRAM reserve**: the store auto-sizes to available RAM − 12 GB, capped at 48 GB
(≈48 GB here), which would leave too little for dev work on this box. The Qwen lanes' analogue
(`resident-budget-gib=28`) is a Flash-Next setting that setup warns about and DeepSeek ignores; the
DeepSeek knob is **`--ssd-cache` in decimal GB** (engine: `budget_bytes = N × 1e9`). The lane passes
`STRATA_SET="ssd-cache=36 reserve-mib=896"` → a 36 GB store cap, leaving about 24 GiB
of the 62 GiB plus reclaimable page cache, with the engine's automatic layer split. Why these
values and the measurements behind them: [`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md).
`--reserve-mib` is the VRAM side and is a **plan-time** input, not a free-VRAM floor (the note has
the measured free-VRAM curve). All are launch settings: change them and restart the lane, no
re-setup. `--ssd-cache 0` turns the store off (every CPU expert read from the SSD per token — do
not).

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

### Deploy record (2026-10-25)

The init (the README command) took about six minutes: the one-time SHA-256 pass over the ~258 GB at
~0.7 GB/s, the 5.8 MB Unsloth metadata shard, the 14 MB pack
(`Strata-data/packs/deepseek-v4.1-flash-ist-das-2.00bpw`), and the 16-token smoke — which passed,
so the gfx110x `strata-dsv41` binary ran its **first kernels on the 7900 XTXs** without a fix.
The lane then started through llama-swap with no setup pass.

The engine log of the lane's start (`state/strata/launch/docker.engine.log`) records the actual
placement:

    engine started: .../strata-dsv41 --serve --pack ... --max-context 262144 --eos 1 --dspark auto
                    --dspark-conf 0.65 --tp 2 --ssd-cache 40 --layer-split auto
    ds4: tensor parallel 2 -> 1: the experts' 256-value blocks split n_ff 2304 no other way (each GPU its own layers)
    ds4: DSpark not loaded: the routed experts (132 GB) are far more than the GPU holds (50 GB) ...
    ds4: RAM store: up to 3893 slots of 10.3 MB (40.0 GB), policy 0, 8 reader threads, unbuffered reads
    ds4: RAM store: 3637 of the profile's hottest experts read in (28.2 GB) in 4.7 s
    ds4: CUDA0 layers [0, 20]: weights 2.96 GB, experts 18.74 GB (2256 of 8064 in VRAM), caches 0.24 GB, 586 MiB free
    ds4: CUDA1 layers [21, 39] + head: weights 2.91 GB, experts 19.15 GB (2097 of 7296 in VRAM), caches 0.10 GB, 592 MiB free
    ds4: 11007 routed experts on the CPU lane (94.9 GB, in place in the model's files), 16 threads

- **Placement**: 4,353 of 15,360 experts (37.9 GB) in VRAM, 11,007 (94.9 GB) on the CPU lane. The
  `--plan` predicted 5,202/45 GB; the engine balances its own layer blocks (0-20 / 21-39) with
  per-card reserves and caches, hence the difference. `--tp 2` was demoted to a plain layer split:
  the IQ2_XXS experts' 256-value blocks cannot be split across two cards. DSpark stays unloaded
  (auto): its verify rounds read 3.3× a token's bytes for no gain at this expert-to-VRAM ratio.
- **RAM**: the 40 GB store is pre-filled with 3,637 hot experts (28.2 GB) at load; at rest the box
  shows ~39 GiB used and ~22–23 GiB available (including reclaimable page cache), so the dev-work
  budget holds. `free` overstates the engine's cost because the Engram table and most CPU experts
  are page cache.
- **API through llama-swap**: `/health` 200; `/v1/models` shows `deepseek-v4.1-flash`, n_ctx
  262144, text-only modalities; a declared tool call returns `get_weather{"city":"Paris"}`
  (`finish_reason: tool_calls`, DSML translated server-side — note `/props` reports
  `supports_tools: false` from the packed template's capabilities, so llama-swap uses the lane's
  declared `tools: true`); `reasoning_effort: "none"` answers without thinking; `max_tokens: -1`
  accepted; streaming SSE ends with `[DONE]`.
- **Timing**: cold start through llama-swap 48.9 s (page cache warm from the init), warm restart
  43.4 s, `docker stop` 1.8 s with VRAM back to 26/232 MiB. The fork's chat-template swap ran on
  the lane's first start ("Chat template: replaced deepseek-v4.1-flash-ist-das-2.00bpw/
  tokenizer/chat_template.jinja …").
- **Cosmetics**: the setup plan prints a spurious "needs more disk space" warning (it compares
  free space with the full 257 GB download even when all files are present; only the 5.8 MB shard
  was fetched). The lane's first start also logs "The model already installed here is somewhere
  else than the folder you gave" — a display quirk of the folder comparison for multi-file
  families; it reuses the install record and does not re-run setup.
- **Ports**: llama-swap expands `${PORT}` in model-ID order, not file order (measured: this ID
  sorts first and got 10001; the Qwen lanes moved to 10008/10009). The per-lane ports are internal;
  only `:8080` is user-facing.
- **Qwen regression after the port move**: the vanilla Qwen lane was started through the swap on
  the new port (10008), said "qwen IQ3_S is already installed with these choices" (no setup pass),
  answered, and stopped in 6.8 s; the pack template already matched the repo copy (the fork's swap
  is silent when equal).

### Speed

The measured curve (20 %/40 % reps, 60 %/85 % calibration prefills), the host RAM/VRAM envelope and
the comparison with the Qwen lane are in [`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md).

0.2.5 (engine 0.1.50) keeps the 6-experts-per-word default here (`--experts`/Speed is the
16 GiB-or-shared-RAM rule; the launch summary shows `Full quality (default on this PC)`), makes the
prompt path length-based and deterministic (chunked pass from 96 tokens, warmed up after ready),
and brings the gfx11 WMMA prompt kernels. On gfx1100 they are opt-in: `STRATA_DS4_HIPK=1` measured
**+18.7 % pp** at ~51K tokens (341.9 → 405.8 t/s, 2 reps; through llama-swap 405.1 t/s), same
answers, decode within noise — the lane now sets that container env. The 0.2.4.2 20 % baseline
(336.6 t/s) matches the 0.1.50 baseline, so the engine itself is neutral. Arms, method and the
start-log lines: [`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md).

### Pending

- Optional: a full standard curve (60 %/85 % tg, out 10 %, reps 2) when hours are available. The
  dsv41 tuner's arms (DSpark forced on, draft block 3, SMT thread counts) were covered by hand on
  2026-10-25 and rejected/neutral — see [`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md).
