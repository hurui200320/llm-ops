# Strata on fedora-tuf

Strata (upstream [`Niko1221/Strata`](https://github.com/Niko1221/Strata)) serves three lanes on the
2× RX 7900 XTX pair: vanilla and SC117 Qwen3.8 Flash-Next (IQ3_S), and IST-DASLab
DeepSeek-V4.1-Flash (2.00 bpw). It runs Docker-only from [our pinned build](../deploy/strata/README.md)
of upstream's 0.2.5 Terminal around the signed-feed 0.1.50 engine, behind the host llama-swap; the
host needs only the `amdgpu` kernel driver — no ROCm, no compiler. The image adds one file the two
Qwen lanes never invoke but the DeepSeek lane runs: the gfx110x `strata-dsv41` build.

Three lanes share the pair through a llama.cpp-style layer split, each in its own container and its
own `/data` tree (the DeepSeek lane uses its own engine and `--tp 2`, demoted to a layer split at
runtime):

| llama-swap ID | Model | Port | Model folder | Data tree |
| --- | --- | ---: | --- | --- |
| `qwen3.8-flash-next-iq3s` | vanilla ISTA-DASLab IQ3_S | 10008 | `/mnt/unenc-xfs/models/strata/ISTA-DASLab-Flash-Next` | `/mnt/unenc-xfs/strata/ISTA-DASLab-Flash-Next` |
| `qwen3.8-flash-next-sc117-iq3s` | SC117 abliterated IQ3_S | 10009 | `/mnt/unenc-xfs/models/strata/sc117-Flash-Next-uncensored` | `/mnt/unenc-xfs/strata/sc117-Flash-Next-uncensored` |
| `deepseek-v4.1-flash-2bpw` | ISTA-DASLab DeepSeek-V4.1-Flash IQ2_XXS (2.00 bpw) | 10001 | `/mnt/unenc-xfs/models/strata/ISTA-DASLab-dsv41f1017` | `/mnt/unenc-xfs/strata/ISTA-DASLab-dsv41f1017` |

The two Qwen lanes use 262144-token context, F16 KV, `PARALLEL=1`, a 28 GiB locked resident budget,
and the same layer split across the cards. All three containers are keyless on the default Docker
bridge with only `127.0.0.1:${PORT}` published; clients and the browser UI reach them through
llama-swap on `:8080`, which loads one lane at a time on the shared pair.

## Pins

| What | Value |
| --- | --- |
| Image | `ghcr.io/hurui200320/llm-ops-strata:0.2.5-rocm-gfx110x`, digest `sha256:0bf409c527a5dd9f008d9392ada2457660b3ec2162acf68df6f4822e9b4a092d`, 1,900,632,969 B on fedora-tuf (`docker inspect .Size`; the dev box reports 1,474,728,281 B for the same digest — storage driver); public GHCR package (no login) |
| Terminal | tag `v0.2.5`, commit `fa663fabc5c2c568d6035b3ae853b38e1762865a`; Terminal/Control `0.2.5`/`0.1.50`; `strata 0.2.5 (build 0.1.50; python 3.12.15)` |
| Engine | `strata-engine-0.1.50-linux-x86_64-hip-rx7900` from the `v0.2.5` release's signed feed, 243,072,991 B, asset sha256 `84698dbf…`, manifest `build_id 6270e2d9d6bf+rocm`, variant `rx7900`, 580 listed files (582 on disk = 580 + `engine-component.json` + our `strata-dsv41`); `engine-component.json` sha256 `43746aaf…` and `BUILD.json` `4b70ff83…` |
| `strata-dsv41` | our gfx110x build from the same `v0.2.5` tag, `strata-dsv41 0.1.50 build fa663fabc`, sha256 `3b6aa77f…`; beside `strata` in the engine folder, not in the manifest by design; **the DeepSeek lane runs it** via `STRATA_DSV41_EXE` (entrypoint fork change 4, verified in-image 2026-10-25) |
| hipBLASLt table | packaged `tuning/gfx1100-hipblaslt-100200.txt`, sha256 `235a26c3…` (873 B), version `100200`; **re-tuned for 0.1.50** (0.1.48 shipped `1bb8cd2e…`, same version) |

The image is our own build of upstream's Terminal (`strata-image.yml`, run 40188533610): the released
engine package plus the one `strata-dsv41` file, so every other byte is upstream's — the signed feed
ships `strata-dsv41` for cuda13 and hip-rx9070 only, hence the gfx110x build. The 0.2.4.2 image stays
pulled for rollback. The engine folder inside the image is
`/opt/strata/components/engine/0.1.50-linux-x86_64-hip-rx7900` (upstream drops the `strata-engine-`
prefix); reference it as `component:strata-engine`, never by path.

## Host and data layout

The pair is 2× RX 7900 XTX (gfx1100, 24 GB each). GPU1 sits behind the Z790 PCH at x4; Strata's
multi-GPU mode is a layer split with hand-off through pinned host RAM, so RCCL is neither possible
nor needed. 62 GiB host RAM, Gen4 NVMe, zram and a swapfile; the locked resident budget, not a
container cap, is the memory control.

```
/mnt/unenc-xfs/
  models/strata/
    ISTA-DASLab-Flash-Next/        vanilla IQ3_S pair + pinned mmproj, 84.53 GB
    sc117-Flash-Next-uncensored/   SC117 IQ3_S pair + own mmproj + unused llama.cpp MTP Q8, 88.91 GB
    ISTA-DASLab-dsv41f1017/        DeepSeek 2.00 bpw set + Engram table + DSpark drafter + mmproj, ~258 GB
  strata/
    ISTA-DASLab-Flash-Next/        vanilla data tree, 7.9 GB (packs/iq3_s 1.5 GB, mtp/rt 6.5 GB, install/, state/, config/, home/, slots/)
    sc117-Flash-Next-uncensored/   SC117 data tree, 7.9 GB (own pack, copied 6.5 GB MTP, lane state)
    ISTA-DASLab-dsv41f1017/        DeepSeek data tree (pack + expert-profile.bin, shared metadata shard)
  llm-ops/                         host checkout of this repo
    deploy/llama-swap.config.yaml
    deploy/strata/entrypoint-local.sh
```

- GGUFs live once, in flat per-lane folders: download, working copy and bind-mount source are the same
  files. The model folder is mounted at `/models`; each lane's data tree is mounted at `/data`. Both
  are plain host directories — bind mounts only, no Docker volumes, so image updates never touch them.
- Each Qwen lane needs its own `/data` tree because both models catalog as tag `iq3_s`. SC117 is not a
  drop-in for the vanilla pack: 144 transplanted tensors changed 95 tensor types to Q8_0 and moved
  1,214/1,223 offsets, so it has a pack built from its own shard 1. Its active mmproj is its own
  907,542,944 B copy (sha256 `2e788f8c…`), not the ISTA pin (`b1a82259…`). The SC117 model folder also
  carries an unused llama.cpp Q8 MTP (4.14 GB, reclaimable).
- `Strata-data/` is the entrypoint's `--data-dir`. `install/`, `state/`, `config/`, `home/` and `slots/`
  sit beside it under `/data`. `PRESET` is left unset, so each lane keeps the image default `docker`
  preset inside its own tree.

## Init / first start

The first start (or a settings change) prepares the model; every later start goes straight to serving.
Setup extracts the chat template into the pack, builds or reuses the native expert pack and MTP layer,
writes the install record and the lane's preset. Init runs outside llama-swap so a long pack/MTP pass
does not hit the proxy's health timeout. The full commands are in
[`deploy/strata/README.md`](../deploy/strata/README.md) ("Run" and "DeepSeek (dsv41) lane").

- `MODEL_PATH` is a folder of GGUFs or a single `.gguf`; `GGUF_DIR` is an alias. Nothing is downloaded
  for the model itself. The prepared pack and MTP layer are written under `/data/Strata-data/`; a later
  start reuses them.
- Install with the default int8 KV: `setup.py` accepts only `int8|q4_0|k8v4` (0.2.5 also accepts `fp16`,
  but a `KV=` value re-runs setup and rewrites the record). F16 KV is applied at launch by
  **`kv=fp16` in `STRATA_SET`**, which replaces the installed `--kv int8` in the engine argv. Do not
  pass `KV=fp16`: it re-runs setup once and rewrites the record (on 0.2.4.2 and earlier it exited 6,
  and `desc_matches()` repeated the failure on every start).
- 0.2.5 changes the engine folder (0.1.48→0.1.50), so the 0.2.4.2 record is dropped and each lane
  re-runs setup once — **no repack and no download** (verified on vanilla 2026-10-25: `Strata-data/`
  stayed at 50 files, only `mtp/tensors/verified.json` was re-stamped, and the record was rewritten
  with the 0.1.50 engine/vision paths; the first request through llama-swap answered after 45.8 s
  including setup, load and 8 generated tokens).
- Chat template: Strata has no `--chat-template-file` flag. Setup extracts the template at pack time
  into `<pack>/tokenizer/chat_template.jinja`, the server reads it per start, and an override is a
  file swap in the pack (it survives re-installs). The Qwen lanes run the repo's
  `templates/qwen3.8/Flash-Next-chat-template.jinja`; the DeepSeek lane names its own
  `templates/deepseek/v4.1-flash-1017-chat-template.jinja`. The entrypoint fork copies the lane's
  `CHAT_TEMPLATE_PATH` over every pack's copy before `strata launch` whenever it is set (the macro
  mounts the repo `templates/` at `/templates:ro` and sets the path per lane). A start where setup
  re-extracts the pack template (a fresh pack, or a pack-format bump) serves the GGUF's own template
  once; the next start restores the repo copy.
- A first setup (no pack yet) makes the fork warn; a repack re-extracts the template after the swap.
  Either way the next start puts the repo template in place.

## llama-swap wiring

[`deploy/llama-swap.config.yaml`](../deploy/llama-swap.config.yaml) defines the shared macro
**`docker-server-strata-rocm`** and appends the three models at the end of `models:` (`${PORT}` is
assigned in **model-ID order** — llama-swap sorts the IDs, not the file — so the DeepSeek lane is
10001 and the two Qwen lanes are 10008/10009 when it was added; measured 2026-10-25). These ports are
internal to the proxy; clients use `:8080`. `CHAT_TEMPLATE_PATH` is per model entry, not in the macro:
each lane names its own template. Host llama-swap runs with `-watch-config`, so entry changes are
picked up live; no environment variable is needed to load a lane.

- Container shape: `--name ${MODEL_ID} --init --rm`, `-p 127.0.0.1:${PORT}:${PORT}`,
  `--entrypoint /entry-local.sh`, `--device /dev/kfd --device /dev/dri`,
  `--group-add 39 --group-add 105` (host gids — see below), `--security-opt label=disable
  --security-opt seccomp=unconfined`, `--ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120`.
- Env: `PORT`, `STRATA_HIPBLASLT_TABLE=on`, `MODEL=IQ3_S`, `CONTEXT=256k`, `GPUS=0,1`, `PARALLEL=1`,
  `MODEL_PATH`, `CHAT_TEMPLATE_PATH`, and the lane's `STRATA_SET` (the DeepSeek lane also sets
  `STRATA_DS4_HIPK=1`). `PRESET` is unset.
- Mounts: the entrypoint fork read-only at `/entry-local.sh`, `/mnt/unenc-xfs/models:/models:z`, the
  repo's `templates/` at `/templates:ro` (Strata never reads it; the fork copies the lane's
  `CHAT_TEMPLATE_PATH` out of it), and the lane's data tree at `/data`. No named volumes, no
  `--memory` cap, no `--network host`.
- Per-model knobs: `cmdStop: docker stop -t 120 ${MODEL_ID}`; `checkEndpoint: /health`;
  `healthCheckTimeout: 600` for the Qwen lanes (cold start 1–3 min, about 30 s warm, and it bounds
  the graceful stop; the DeepSeek lane overrides it to 1200); `concurrencyLimit: 20` (llama-swap
  queue depth; the engine still has one slot). `capabilities` is declared rather than probed because
  Strata's `/v1/models` omits `owned_by`. The `:npt` alias uses
  `chat_template_kwargs.preserve_thinking: false`.
- Fork: `deploy/strata/entrypoint-local.sh` is upstream's entrypoint plus four changes — pass
  `--no-api-key` when no key is set, suppress upstream's false "Strata made one" warning, replace each
  pack's `tokenizer/chat_template.jinja` with `CHAT_TEMPLATE_PATH` when it is set, and export
  `STRATA_DSV41_EXE` from `/opt/strata/components/engine/*/strata-dsv41` when unset (the DeepSeek
  engine, which this image's Terminal never finds on its own; see the DeepSeek section). `launch`
  stores no key, so llama-swap keeps `apiKeys: []`; the LAN reaches Strata only through the swap, and
  `/upstream/<model>/` UIs work. Re-diff the fork on every image bump; a missing bind source becomes a
  root-owned directory and kills the container.

Current `STRATA_SET` (identical on the two Qwen lanes; the DeepSeek lane passes only
`ssd-cache=36 reserve-mib=896` plus the container env `STRATA_DS4_HIPK=1`):

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
3591/3591, GPU caches about 31.3 GiB and a 22.4 GiB RAM copy after `trim-stage-weights`. Tuning
history and rejects: [`notes/qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md).

## Operating notes

- **Start/stop**: a cold load takes 1–3 min, a warm start about 30 s. SIGTERM drains, unloads and
  frees VRAM via `strata launch --stop`; `docker stop -t 120` is sized for that.
- **Upgrades**: pull and pin the new image, back up `install/strata-iq3_s.json` and
  `config/strata/launch-presets/docker.json`, make sure no Strata model is loaded, re-diff the
  entrypoint fork, then let llama-swap reload or start the vanilla lane first. See "Upgrade history".
- **Vision token cap**: the server and the reader clamp image tokens by default when
  `vision.max_tokens` is absent (CPU reader 300, GPU reader 1024; #3690). The lane's install record
  carries the reader settings setup wrote; the 2026-10-23 smoke verified a 640×360 picture.
- **Never** pass `KV=`; use `kv=` inside `STRATA_SET`. 0.2.5's setup accepts `fp16`, but `KV=fp16`
  re-runs setup once and rewrites the record. `desc_matches()` compares the KV choice with the
  installed config, so the lanes stay installed with int8 and `kv=fp16` appends `--kv fp16` to the
  engine argv (verified 2026-10-25).
- **Never** pass `--slot-save-path`: session files are refused on a layer split. Conversation parking
  (`conversation-cache-*`) is the working cache mechanism.
- **`STRATA_HIPBLASLT_TABLE` must be a container env var.** Inside `STRATA_SET` it lands in the
  preset's `env` and is silently ignored. Without it prompts run plain hipBLAS; the start log should
  say `prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)`.
- **`--group-add` takes numeric host gids.** Docker resolves names against the image's `/etc/group`,
  which has no `render` and maps `video` to 44; on fedora-tuf the host values are 39 (video) and
  105 (render).
- **`PARALLEL` unset is not one request.** The image default fits the card and passes `--batch 3` on
  24 GB. Keep `PARALLEL=1`.
- **No memory cap.** Upstream advises none; the engine plans from host meminfo and the entrypoint
  warns on `memory.max`. The 28 GiB resident budget is the control.
- **No volumes contract.** A missing shard fails fast by name at `check_shards`/`GGUF_DIR`, but a
  missing mmproj is not a hard failure — setup would silently download about 907 MB into the data
  tree. Keep each lane's GGUFs and mmproj co-located in its flat folder. `--gguf-dir` scans one flat
  folder: exact published name first, else the `*-00001-of-*` glob.
- **Integrity**: per-file `.sha256` sidecars are bare 64-character hex digests (no filename, no
  newline). Verify with
  `printf '%s  %s\n' "$(cat f.gguf.sha256)" f.gguf | sha256sum -c -`. Setup pins no hashes for the
  ISTA files; the sidecar pass is the integrity gate.
- **Chat template**: the Qwen lanes run the repo's
  `templates/qwen3.8/Flash-Next-chat-template.jinja` (byte-identical to HF
  `Qwen/Qwen3.8-Flash-Next` main on 2026-10-24, 8,952 B), not the one embedded in the GGUFs. To
  update: `templates/fetch-templates.sh`, commit, `git pull` in `/mnt/unenc-xfs/llm-ops`, restart the
  lane. Nothing is repacked and no GGUF is re-downloaded. Verified on fedora-tuf 2026-10-24: both
  packs were rewritten right after the pull, a deliberately diverged pack was replaced at the next
  start, and a short chat, a declared tool call (`get_weather{"city":"Paris"}`) and the `:npt` alias
  all answered through llama-swap.
- **API key**: the swap lanes are keyless and have no `/data/state/api-key`. A manual run with
  `API_KEY` set still creates and uses a key.

## Upgrade history

| Upgrade | Engine | First start | Verification |
| --- | --- | --- | --- |
| 0.2.2 → 0.2.3 (upstream image) | 0.1.47 → 0.1.48 | record dropped; setup once; no repack/download | within noise; SC117 swapped in with its own mmproj |
| 0.2.3 → 0.2.4.2 (own image) | 0.1.48 unchanged | record kept; straight to serving (no setup) | within noise (see Flash-Next note) |
| 0.2.4.2 → 0.2.5 (own image) | 0.1.48 → 0.1.50 | record dropped; setup once; no repack/download; hipBLASLt table re-tuned | neutral; DeepSeek WMMA kernels enabled (see dsv41 note) |

0.2.5 engine changes: the Qwen engine adds CPU-row kernels (AVX-2 on this host — no AVX-512) and the
re-tuned hipBLASLt table; `strata-dsv41` adds `--experts` (6 here), the length-based deterministic
prompt path with background warm-up, fewer launches per layer with store-hit/miss overlap, and the
opt-in gfx11 WMMA kernels. Qwen spot-check numbers, method and the 0.2.5 table A/B:
[`notes/qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md). DeepSeek engine behaviour and WMMA:
[`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md).

## DeepSeek-V4.1-Flash lane (dsv41)

Deployed 2026-10-25; the measured speed curve and tuning record are in
[`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md). It serves the IST-DASLab **2.00 bpw** set
(IQ2_XXS) of DeepSeek-V4.1-Flash (1017B-parameter MoE, 40 layers × 384 experts, top-6) on the
image's own `strata-dsv41` engine (0.1.50 since the 0.2.5 image).

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

`strata-dsv41` is the image's own gfx110x build (0.1.50; pins above), placed beside `strata` in the
engine package folder — the shape an upstream `DSV41=1` rx9070 package has. That shape is auto-found
only by a desktop install (`ROOT/engine` *is* the package folder there). **In this image it is not
found**: the Terminal's launch runs setup with
`STRATA_CONTROL_INSTALL=1` / `STRATA_INSTALL_DIR=/data/install`, so `dsv41_exe()` looks in
`/data/install/engine-dsv41/` and `/opt/strata/terminal/engine/`, and the launch's
`dsv41_engine_found()` probes only `engine-dsv41/` stores — verified in-image 2026-10-25. Without a
fix, first setup stops at "DeepSeek's engine is not available for this computer yet" (the signed
feed ships `strata-dsv41` for cuda13 and hip-rx9070 only).

Fix: **entrypoint fork change 4** exports `STRATA_DSV41_EXE` from
`/opt/strata/components/engine/*/strata-dsv41` when it is unset (an explicit `-e` still wins; inert
for the Qwen lanes, whose configs are not `arch: deepseek41`). Verified in-image 2026-10-25 with
0.1.48 (`strata launch ... --plan` reported `DeepSeek engine: installed, 0.1.48`; without the fork
the plan warns that setup would have to download the engine). The 0.2.5 lanes resolve it the same
way: the engine starts from the 0.1.50 package path and the launch's Speed line (`Full quality`) is
gated on `dsv41_engine_version(exe) >= 0.1.50`, which it reads from the package folder.
`strata-dsv41 --version` runs in the image; the binary had **never executed a kernel on gfx1100
hardware** before this lane's first deploy (2026-10-25) — setup's 16-token smoke was that test, and
the 0.1.50 rebuild repeats it on the 0.2.5 setup pass.

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
`STRATA_SET="ssd-cache=36 reserve-mib=896"` → a 36 GB store cap, leaving about 24 GiB of the 62 GiB
plus reclaimable page cache, with the engine's automatic layer split. Why these values and the
measurements behind them: [`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md). `--reserve-mib` is
the VRAM side and is a **plan-time** input, not a free-VRAM floor (the note has the measured free-VRAM
curve). All are launch settings: change them and restart the lane, no re-setup. `--ssd-cache 0`
turns the store off (every CPU expert read from the SSD per token — do not).

### Context, KV, parallel, vision

- **Context 262144 (256k)**: `--context 256k` → `--max-context 262144`. Upstream checked the engine
  at 4K/8K only; setup warns "longer is kept as you chose, but untested" and keeps it. Fallback
  ladder: 128k, then 64k (the engine default is 8192).
- **No KV setting exists**: `strata-dsv41` has no `--kv` flag — its cache is its own representation
  (~17 MB per 6.5K tokens, ≈0.7 GB at 256k). `kv=fp16` in `STRATA_SET` would append `--kv fp16` to
  the engine argv and kill it; `KV=`/`resident-budget-gib` are Flash-Next settings that setup rejects
  for deepseek41.
- `--parallel` and `--vision` are ignored with a warning in this engine version: one request at a
  time, text only. The lane declares `capabilities: in: [text]` and passes neither.
- Thinking is on per the template; `reasoning_effort: "none"` or
  `chat_template_kwargs.enable_thinking: false` turns it off. No `:npt` alias yet (that keyword is
  `preserve_thinking`, a Qwen-template thing).

### Init / first start

The first start does the setup work — run it outside llama-swap (same pattern as the Qwen lanes; the
full command is in [`deploy/strata/README.md`](../deploy/strata/README.md)):

1. one-time SHA-256 pass over the ~257 GB of present files (several minutes on the NVMe);
2. the 5.8 MB Unsloth metadata shard download;
3. pack build (`Strata-data/packs/deepseek-v4.1-flash-ist-das-2.00bpw/`: tensor map, hash tables,
   tokenizer, the GGUF's own chat template) + `expert-profile.bin`; a few MB to ~6 GB of disk;
4. `strata-dsv41 --pack-info` (every file present), then the 16-token smoke whose answer must name
   Paris — the first kernel run on gfx1100 (allowed up to 5400 s; the cold load reads the experts).

Then stop the container and let llama-swap start the lane. Later starts take a few minutes (the CPU
lane's experts are read first), hence the lane's 1200 s health timeout.

### Deploy record (2026-10-25)

The init took about six minutes: the one-time SHA-256 pass over the ~258 GB at ~0.7 GB/s, the
5.8 MB Unsloth metadata shard, the 14 MB pack
(`Strata-data/packs/deepseek-v4.1-flash-ist-das-2.00bpw`), and the 16-token smoke — which passed,
so the gfx110x `strata-dsv41` binary ran its **first kernels on the 7900 XTXs** without a fix. The
lane then started through llama-swap with no setup pass.

The engine log (`state/strata/launch/docker.engine.log`) records the actual placement:

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
  per-card reserves and caches. `--tp 2` was demoted to a plain layer split: the IQ2_XXS experts'
  256-value blocks cannot be split across two cards. DSpark stays unloaded (auto): its verify rounds
  read 3.3× a token's bytes for no gain at this expert-to-VRAM ratio.
- **RAM**: the first deploy's 40 GB store is pre-filled with 3,637 hot experts (28.2 GB) at load; at
  rest the box shows ~39 GiB used and ~22–23 GiB available (including reclaimable page cache), so the
  dev-work budget holds. `free` overstates the engine's cost because the Engram table and most CPU
  experts are page cache.
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

### Speed

The measured curve (20 %/40 % reps, 60 %/85 % calibration prefills), the host RAM/VRAM envelope and
the tuning arms are in [`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md).

0.2.5 (engine 0.1.50) keeps the 6-experts-per-word default here (`--experts`/Speed is the
16 GiB-or-shared-RAM rule; the launch summary shows `Full quality (default on this PC)`), makes the
prompt path length-based and deterministic (chunked pass from 96 tokens, warmed up after ready), and
brings the gfx11 WMMA prompt kernels. On gfx1100 they are opt-in: the lane sets
`STRATA_DS4_HIPK=1`; the measured pp gain, same-answer check and A/B are in
[`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md). The 0.2.4.2 20 % baseline matches the 0.1.50
baseline, so the engine itself is neutral.
