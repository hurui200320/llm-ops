# Strata on fedora-tuf

Strata (upstream [`Niko1221/Strata`](https://github.com/Niko1221/Strata)) serves Qwen3.8 Flash-Next
(125B-class MoE, IQ3_S GGUF) on the 2× RX 7900 XTX pair. It runs from upstream's official image,
Docker-only, behind the host llama-swap; the host needs only the `amdgpu` kernel driver — no ROCm,
no compiler.

Two lanes share the pair through a llama.cpp-style layer split, each in its own container and its
own `/data` tree:

| llama-swap ID | Model | Port | Model folder | Data tree |
| --- | --- | ---: | --- | --- |
| `qwen3.8-flash-next-iq3s` | vanilla ISTA-DASLab IQ3_S | 10007 | `/mnt/unenc-xfs/models/strata/ISTA-DASLab-Flash-Next` | `/mnt/unenc-xfs/strata/ISTA-DASLab-Flash-Next` |
| `qwen3.8-flash-next-sc117-iq3s` | SC117 abliterated IQ3_S | 10008 | `/mnt/unenc-xfs/models/strata/sc117-Flash-Next-uncensored` | `/mnt/unenc-xfs/strata/sc117-Flash-Next-uncensored` |

Both use 262144-token context, F16 KV, `PARALLEL=1`, a 28 GiB locked resident budget, and the same
layer split across the cards. The containers are keyless on the default Docker bridge with only
`127.0.0.1:${PORT}` published; clients and the browser UI reach them through llama-swap on `:8080`.

Both lanes serve through llama-swap (one loaded at a time) on image 0.2.3 (engine 0.1.48),
deployed 2026-10-23 (commit `cae796b`); the lane tuning is from 2026-10-22 and carried over
unchanged. 0.2.3 spot-check (vanilla, through llama-swap): 60 % pp/tg 2077/66.0, 85 % 2132/63.3
(0.2.2: 2017/67.4 and 2174/66.5 — within noise). The full runbook is
[`deploy/strata/README.md`](../deploy/strata/README.md); speed and quality measurements are in
[`notes/qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md).

## Pins

| What | Value |
| --- | --- |
| Image | `ghcr.io/niko1221/strata:0.2.3-rocm-gfx110x`, digest `sha256:4dd91b54cdf64870b837d00fb031c9b0531d6ea6a9b6fdf72d506912d3029109`, 1,847,098,151 B |
| Terminal | tag `v0.2.3`, commit `5c8dd322528cc84f68f461ad54a506b9da8fa8d4`; `strata 0.2.3 (build 0.1.48; python 3.12.15)` |
| Engine | `strata-engine-0.1.48-linux-x86_64-hip-rx7900`, 242,858,823 B, sha256 `13bcf38c…`, manifest `build_id 6f8800aefe27+rocm`, variant `rx7900`, 581 files (`engine-component.json` sha256 `59a2ac74…`, byte-identical to the image) |
| hipBLASLt table | packaged `tuning/gfx1100-hipblaslt-100200.txt`, sha256 `1bb8cd2e…`, version `100200` (unchanged from 0.2.2, verified 2026-10-23) |

The image is upstream's published package (public, no login). We do not build or patch images
locally; a new upstream image is deployed as a new pin. The engine folder inside the image is
`/opt/strata/components/engine/0.1.48-linux-x86_64-hip-rx7900` (upstream drops the
`strata-engine-` prefix); reference it as `component:strata-engine`, never by path.

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
  -v /mnt/unenc-xfs/llm-ops/deploy/strata/entrypoint-local.sh:/entry-local.sh:ro \
  -v /mnt/unenc-xfs/strata/ISTA-DASLab-Flash-Next:/data \
  -v /mnt/unenc-xfs/models:/models:z \
  -e MODEL=IQ3_S -e CONTEXT=256k -e GPUS=0,1 -e PARALLEL=1 \
  -e MODEL_PATH=/models/strata/ISTA-DASLab-Flash-Next \
  -e STRATA_SET="kv=fp16 vram-reserve-mib=1536 vram-reserve-later-mib=768 resident-budget-gib=28 prompt-cache-spread=on conversation-cache-mib=8192 conversation-cache-slots=8 pipeline-windows=2 layer-split=26 trim-stage-weights=on spec-min-p=0.7 STRATA_SCORES_THREAD=1" \
  ghcr.io/niko1221/strata:0.2.3-rocm-gfx110x
docker logs -f strata-init
```

Stop the container when setup finishes; the swap lane then reuses the prepared tree.

- `MODEL_PATH` is a folder of GGUFs or a single `.gguf`; `GGUF_DIR` is an alias. Nothing is
  downloaded for the model itself. The prepared pack and MTP layer are written under
  `/data/Strata-data/`; a later start reuses them.
- Install with the default int8 KV: `setup.py` accepts only `int8|q4_0|k8v4`. F16 KV is applied at
  launch by **`kv=fp16` in `STRATA_SET`**, which replaces the installed `--kv int8` in the engine
  argv. Do not pass `KV=fp16` — `strata launch` forwards it to setup, which exits 6, and
  `desc_matches()` repeats the failure on every start.
- 0.2.3's first start drops the 0.2.2 install record (it names the 0.1.47 engine/vision paths),
  re-runs setup once and reuses the existing pack and MTP. No repack and no engine/mmproj
  download. Both lanes go through this; the SC117 setup selects its own mmproj from the model
  folder. A warm first start took ~40 s in the 2026-10-23 smoke (≈30 s of it the model load);
  the same smoke verified chat, a declared-tool call, `max_tokens:-1` and 640×360 vision, and a
  warm second start needed no setup pass (stop 2.4 s, exit 0, VRAM freed).
- The vanilla pack needs no chat-template override. The template is extracted at pack time into
  `<pack>/tokenizer/chat_template.jinja` and read per start; an override is a file swap in the pack
  (there is no `--chat-template-file` flag), and it survives re-installs.

## llama-swap wiring

[`deploy/llama-swap.config.yaml`](../deploy/llama-swap.config.yaml) defines the shared macro
**`docker-server-strata-rocm`** and appends the two models at the end of `models:` (`${PORT}` is
assigned in config order, hence 10007/10008). Host llama-swap runs with `-watch-config`, so entry
changes are picked up live; no environment variable is needed to load a lane.

- Container shape: `--name ${MODEL_ID} --init --rm`, `-p 127.0.0.1:${PORT}:${PORT}`,
  `--entrypoint /entry-local.sh`, `--device /dev/kfd --device /dev/dri`,
  `--group-add 39 --group-add 105` (host gids — see below), `--security-opt label=disable
  --security-opt seccomp=unconfined`, `--ulimit memlock=-1:-1 --shm-size 2g
  --stop-timeout 120`.
- Env: `PORT`, `STRATA_HIPBLASLT_TABLE=on`, `MODEL=IQ3_S`, `CONTEXT=256k`, `GPUS=0,1`,
  `PARALLEL=1`, `MODEL_PATH`, and the lane's `STRATA_SET`. `PRESET` is unset.
- Mounts: the entrypoint fork read-only at `/entry-local.sh`, `/mnt/unenc-xfs/models:/models:z`,
  and the lane's data tree at `/data`. No named volumes, no `/templates` (Strata never reads it),
  no `--memory` cap, no `--network host`.
- Per-model knobs: `cmdStop: docker stop -t 120 ${MODEL_ID}`; `checkEndpoint: /health`;
  `healthCheckTimeout: 600` (cold start 1–3 min, about 30 s warm, and it bounds the graceful stop);
  `concurrencyLimit: 20` (llama-swap queue depth; the engine still has one slot).
  `capabilities` is declared rather than probed because Strata's `/v1/models` omits `owned_by`.
  The `:npt` alias uses `chat_template_kwargs.preserve_thinking: false`.
- Keyless boundary: `deploy/strata/entrypoint-local.sh` is upstream's entrypoint plus two changes —
  pass `--no-api-key` when no key is set, and suppress upstream's false "Strata made one" warning.
  `launch` stores no key, so llama-swap keeps `apiKeys: []`; the LAN reaches Strata only through
  the swap, and `/upstream/<model>/` UIs work. Re-diff the fork on every image bump; a missing bind
  source becomes a root-owned directory and kills the container.

Current `STRATA_SET` (identical on both lanes):

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
  entrypoint fork, then let llama-swap reload or start the vanilla lane first. The 0.2.2→0.2.3
  pack and MTP are compatible without a repack.
- **0.2.3.1 is a Control-only hotfix and has no docker image (as of 2026-10-23).** The 0.2.3 image
  bundles Control 0.1.48; 0.2.3.1's Control 0.1.48.1 fixes `BUILD.json` archs parsing on the
  Windows NVIDIA package (#3810) and an AMD engine-recognition check on the Windows HIP path. The
  container runs setup with `STRATA_CONTROL_INSTALL=1` and resolves the engine through
  `strata_control/engine_component.py`, so neither fix is on this lane's path, and the 0.1.48
  engine packages are identical in both releases. If upstream publishes `0.2.3.1-rocm-gfx110x`,
  prefer it — a drop-in tag change.
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
- **Chat template**: no override needed for either lane. The pack's extracted
  `tokenizer/chat_template.jinja` is what the server reads; a custom template means replacing that
  file, not mounting `/templates`.
- **API key**: the swap lanes are keyless and have no `/data/state/api-key`. A manual run with
  `API_KEY` set still creates and uses a key.
