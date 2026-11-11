# Maya (AMD / HIP) deployment

The Maya image is **our own build** (pushed by the workflow, tag `1.1.59-rocm-gfx1100`), made by
[`.github/workflows/maya-image.yml`](../../.github/workflows/maya-image.yml). It carries Project Maya `v1.1.59`
with its **GLM-5.3-Flash engine** (`maya`), its **DeepSeek-V4.1-Flash engine** (`maya-ds4`) and the **GPU vision
encoder** (`maya-vision`), compiled for **gfx1100** (RX 7900 XT / XTX) inside AMD's ROCm `7.2.1` dev container, plus
the installer's Python environment (`.venv`) and the "already compiled" stamps `maya.py` checks. The host needs only
the `amdgpu` kernel driver: no ROCm, no compiler. The image is ~30 GB uncompressed (the ROCm SDK is most of it);
our own layers are ~0.65 GB.

We build our own image because upstream publishes none: `maya.py` compiles the engines on the user's machine at
first start. Our image does that compile once, the way the installer does it, so the first start only downloads the
model (or uses files you already have), builds its pack and serves. Historical context, pins and verification records:
[notes/maya.md](../../notes/maya.md).

## Pins

Current image: `ghcr.io/hurui200320/llm-ops-maya:1.1.59-rocm-gfx1100` (Maya `v1.1.59`, GLM + DeepSeek engines,
`gfx1100`). The workflow `env:` block pins the build inputs; the image digest of each build is in
[notes/maya.md](../../notes/maya.md).

| Input | Pin |
| --- | --- |
| Maya source | `mw00/project-maya` tag `v1.1.59` |
| llama.cpp | commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` (maya's `setup.py` pin; gated by the workflow) |
| Base image (build + runtime) | `docker.io/rocm/dev-ubuntu-24.04:7.2.1-complete` |
| GPU arch | `gfx1100` only |
| hipBLASLt tuning table | `tools/hip/gfx1100-glm-hipblaslt-100202.txt` (ROCm 7.2.1's hipBLASLt 1.2.2) |
| ggml ISA | `STRATA_PORTABLE=ON` (AVX2 baseline: `GGML_NATIVE=OFF` + AVX/AVX2/FMA/F16C/BMI2; asserted in the build) - the image must run on every host maya accepts, not just the builder |

The base is `-complete` on purpose: `maya.py`'s `check_hip_pc` runs at **every start** and requires `hipcc` ≥ 7,
ROCm's clang and `libhipblas` inside the environment, so a libs-only runtime image would refuse to start. The
hipBLASLt table matches the SDK in the image; the engine accepts only its own arch + hipBLASLt version and falls
back to plain hipBLAS otherwise (the engine log says which).

## Run

Host prerequisites (once): the `amdgpu` kernel driver, `/dev/kfd` and `/dev/dri`, and the host's `video`/`render`
group ids (Docker resolves `--group-add` names against the **image's** `/etc/group`, not the host's — pass numbers,
e.g. `getent group video render` → 39/105 on fedora-tuf):

```sh
docker run -d --name maya --init --restart unless-stopped \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt label=disable --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
  -p 127.0.0.1:8080:8080 \
  -v /mnt/unenc-xfs/maya/<model name>:/data \
  -v /mnt/unenc-xfs/models/maya/<shard folder>:/data/model:ro \
  -e PACK_BUILD=1 \
  -e PORT=8080 -e GPU=0 \
  ghcr.io/hurui200320/llm-ops-maya:1.1.59-rocm-gfx1100
```

`1.1.59-rocm-gfx1100` is the image's only tag; `ghcr.io/hurui200320/llm-ops-maya:latest` is not published.

- One **data tree per model** (the strata pattern): `/mnt/unenc-xfs/maya/<model name>` holds everything the lane
  derives — the pack, the vision tower, the installed configs (`/data/config/`) — so lanes never share state. The
  **GGUF shards stay in the production folder** (`/mnt/unenc-xfs/models/maya/…`) and are bind-mounted **read-only**
  at `/data/model`: the engine opens them in place through the pack's `model_dir` — no hard links, no copies. The
  tower and the pack are built into the writable `/data` (see "First start").
- Why the shards are mounted **inside** `/data` and not a shared `/models` mount like the strata lanes use: the
  pack records the vision tower's file relative to the model dir (`pack_gguf.py` writes `../vision/vision.bin`
  into the pack's file list; the engine resolves that path against `model_dir` at every load — there is no engine
  flag for it). So a **writable** `vision` folder must sit beside the model dir inside the container. The shared
  models folder cannot host it: the mount is read-only (the production tree is shared state) and `vision.bin` is
  built from the mmproj, i.e. derived state that belongs in the lane's data tree. `/data/model` + `/data/vision`
  are siblings, so `../vision` resolves into `/data`.
- The container publishes `127.0.0.1:${PORT}` only (same layout as the Strata lane); the LAN reaches it through
  llama-swap. It listens on `0.0.0.0` inside the container so the port mapping works. Without `API_KEY` anyone who
  can reach the published port can use it.
- `--init` reaps the engine process tree; `--stop-timeout 120` lets a start/stop drain instead of SIGKILL;
  `--ulimit memlock=-1` is what the expert cache's pinned RAM tier needs (native Maya asks for it too);
  `--shm-size 2g` keeps the prompt staging off the default 64 MB `/dev/shm`.

### First start

With `PACK_BUILD=1` and the shard folder mounted at `/data/model`, the first start **self-provisions**: the
entrypoint builds whatever is missing into the writable data tree and then serves (a restart only stats):

1. **Vision tower** → `/data/vision/` (`vision.bin` + `vision.json`): `tools/ds4/vision_pack.py --mmproj` reads the
   mount's `mmproj-*.gguf` (~1 GB written). Skipped when `vision.bin` is already there; delete it to force
   a rebuild.
2. **Pack** → `/data/pack/maya-ds4/` (`ds4.txt`, `tensors.txt`, `engram.bin`, `tokenizer/`): 
   `tools/ds4/pack_gguf.py --model /data/model --out /data/pack/maya-ds4 --vision /data/vision --shards …`
   reads the shards' headers only (seconds — no copy of the ~240 GB). The entrypoint passes every `.gguf` in the
   mount except `mmproj-*.gguf` (the tower's source, architecture `clip`, which `pack_gguf.py` would otherwise
   take as a shard; ISTA's sixth shard publishes under the Engram name and is included). Skipped when `ds4.txt`
   is there; delete the folder to force a rebuild (e.g. after a pack-tools change upstream).
3. The entrypoint then normalizes every pack's `model_dir` to `/data/model` (a pack built elsewhere — e.g. on the
   host — still resolves inside the container) and applies the picture budget before the first load, so even a
   fresh pack serves with `DS4_VISION_TOKENS` right away.

Without `PACK_BUILD=1` a missing pack or tower stops the start with the fix instead of the engine's load error.

The SETUP path below is for **Maya-managed quants** (a Maya quant id in `MODEL` — GLM Maya-S and friends, downloaded
or given via `--gguf-dir`); the DeepSeek-style lanes above never use it (their config comes from `CONFIG_SRC` and
their pack from `PACK_BUILD`):

```sh
# Maya-managed quants: files you already have (GGUF/EXL3), or let the container download
docker run -d --name maya-setup --init --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt label=disable --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --shm-size 2g \
  -v /mnt/unenc-xfs/maya/<model name>:/data \
  -e GPU=0 -e CONTEXT=262144 \
  ghcr.io/hurui200320/llm-ops-maya:1.1.59-rocm-gfx1100 \
  --setup --backend hip --yes --no-start \
  --model glm-maya-s --gguf-dir /data/models/GLM-5.3-Flash-Maya-S \
  --host 0.0.0.0 --port 8080
docker logs -f maya-setup    # wait for the setup to finish, then it exits
```

- **Any arguments are passed to `maya.py` unchanged** (`--check`, `--setup`, `--bench`, `--report`, `--calibrate`…):
  the image only wraps the start.
- `--gguf-dir` takes the folder with all GGUF files (or the first `.gguf` of a split model); the folder must be
  writable because the pack is written inside it. DeepSeek: `--model deepseek-maya-s --model-dir <folder>`.
- To let the container download the model instead (a Maya quant id in `MODEL`, ~96.5 GB for GLM Maya-S):
  `-e SETUP=1 -e DOWNLOAD=1 -e MODEL=glm-maya-s` with no arguments.
- `--no-start` makes the setup exit when it is done; the serve container then reuses the prepared tree. The setup
  copies its config to `/data/config/` (Maya writes it into its own folder, which is the container layer).
- Pictures: the GLM vision encoder is built into the image; the setup downloads its ~1.14 GB of files into the model
  folder. `--no-vision` leaves them out.

### Serve

```sh
docker run -d --name maya ... ghcr.io/hurui200320/llm-ops-maya:1.1.59-rocm-gfx1100
```

With no arguments the entrypoint serves the newest `/data/config/maya-*-hip.json` (`--backend hip --config … --host
0.0.0.0 --port $PORT`), which is what the dashboard on `/` and the OpenAI/Anthropic API on `/v1` use. `GET /health`
is the readiness probe (the healthcheck waits up to 30 min for a first load).

## Environment (no-argument start)

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | `8080` | dashboard + API port inside the container |
| `HOST` | `0.0.0.0` | bind address inside the container |
| `GPU` / `GPUS` | `0` / unset | one card (`--gpu N`) or a layer split (`--gpus 0,1`) |
| `MODEL` | `glm-maya-s` | setup id, used only when `SETUP=1` and no config exists |
| `CONTEXT` | Maya's default | context for that setup (e.g. `262144`) |
| `MODELS_DIR` | `/data/models` | where `SETUP`-installed (Maya-managed) models live |
| `API_KEY` | unset | require this key from API clients |
| `SETUP` / `DOWNLOAD` | `0` / `0` | run the setup when no config exists / download the model in it |
| `PACK_BUILD` | `0` | build the vision tower (`/data/vision`) and the pack (`/data/pack/maya-ds4`) from the shards mounted at `/data/model` when missing — the fresh-data-tree self-provisioning of the DeepSeek-style lanes; without it a missing pack or tower stops the start with the fix |
| `DS4_VISION_TOKENS` | unset | DeepSeek picture budget (prompt tokens), written into every `maya-ds4` pack's `ds4.txt` (`vision.max_image_tokens`) at every start; the tower's hard maximum is 1140 (10,240-patch attention) and a larger value is clamped with a warning. Unset: the pack keeps its own value (`pack_gguf.py` writes 1024) |
| `DS4_VISION_MIN_PIXELS` | unset | DeepSeek: upscale a small picture to at least this many pixels (the pack's `vision.min_pixels`). Unset: the pack keeps its own value |

## Data layout

One tree per model (the strata pattern). The DeepSeek lane mounts its shards as a **folder**; the GLM lane
binds its shards **per file** (the GLM engine has no `model_dir` key: it opens the shards as siblings of its
pack — `src/core/glm_model.cu` reads `<pack>/../<shard name>`).

```
DeepSeek lane (maya-ds4)
/data/                                   one tree per model (the strata pattern)
  model/                                 READ-ONLY mount of the production shard folder
                                         (the GGUFs the engine reads in place; an empty dir on the host -
                                         it only has content inside the container)
  vision/                                the vision tower, built from the mount's mmproj (vision.bin, vision.json)
  pack/maya-ds4/                         the pack: ds4.txt (geometry + model_dir /data/model), tensors.txt,
                                         engram.bin, tokenizer/, the engine's expert profile and caches
  config/maya-<quant>-hip.json           the lane's installed config, copied from CONFIG_SRC at every start
                                         (newest wins; one lane per folder)
  models/                                only SETUP-installed (Maya-managed) models land here

GLM lane (maya)
/data/                                   one tree per model
  GLM-…-0000X-of-00003.gguf              READ-ONLY per-file binds beside pack/, where the engine opens them
                                         (Docker leaves 0-byte root-owned placeholders on the host; the mounts
                                         cover them while the lane runs - do not delete or repurpose them)
  pack/                                  the pack: index.txt, native_experts.txt, dense.bin (0.88 GiB), tokenizer/
  pack/slots/                            the GLM engine's conversation slots (disk, 4 slots / 16 GB, kept across runs)
  vision/                                mmproj + vocab GGUF, downloaded by the setup (maya-vision encodes)
  config/maya-<quant>-hip.json           the lane's installed config, copied from CONFIG_SRC at every start
                                         (newest wins; one lane per folder)
```

Per-start settings (context, expert cache…) live in the config; the dashboard edits it. Logs go to
`/opt/maya/maya-<quant>-hip.log` in the container layer (mount nothing there; use `docker logs` for the server).

### Configs from the repo (lanes)

`deploy/maya/configs/` holds version-controlled configs for lanes, **one subfolder per lane** named after the
llama-swap model id (`dsv41f1017-iq2xxs/`, `glm53f-iq2s-derisked/`, `glm53f-iq2s/`), each holding one `maya-*-hip.json`
so the entrypoint's glob finds it. Each lane mounts **its own** folder read-only and points `CONFIG_SRC` at it (the
llama-swap macro mounts none): the entrypoint serves the newest copied JSON, so a shared folder would let one
lane's config take over the other on its next restart. The entrypoint copies the JSON into `/data/config` at
**every** start, so the repo file is the declared state (the equivalent of the strata fork's `STRATA_SET` env)
and dashboard edits last until the next lane restart. Never mount a repo config writable straight into
`/data/config`: maya.py touches (and may rewrite) the config on every start, which would dirty the checkout. The
DeepSeek lane's config is `configs/dsv41f1017-iq2xxs/maya-deepseek-2bpw-hip.json` (512k, `--ssd-cache 36
--reserve-mib 1664 --suffix-draft 0`, env `DS4_INDEX_TC=1 DS4_CHUNK_GFX11=16384 DS4_RING_MB=128 DS4_HC_STREAM=1
DS4_HC_MIX_SPLIT=2 DS4_ROUTE_FUSE=1 DS4_SPLIT=18,22 MAYA_DS4_SLOT_KEEP=1 MAYA_DS4_SLOT_CAP=8
MAYA_DS4_SLOT_RAM_GB=6`; rationale, tuning arms and numbers: notes/deepseek/v41f1017.md 20261104/20261105).

The engine knobs on that lane, in one line each (they exist only as engine environment, not config keys):

| Knob | Why |
| --- | --- |
| `DS4_SPLIT=18,22` | layer split (maya CUDA0 gets layers [0,17], CUDA1 [18,39]+head). The auto planner balances *streamed expert bytes* and assumes both links are equal; on this pair the monitor card also carries the desktop's ~460 MiB, so it ends 21/19 and pays a lot for the marginal experts. Spreading towards the x4 card frees expert room on the monitor card: 4,559 experts in VRAM (auto 4,499), pp +41 % at 40 % and +44 % at 85 % |
| `DS4_CHUNK_GFX11=16384` | raises gfx11's hard 4096 chunk cap so the engine's **borrowed** 16K prompt chunk engages: the chunk's row buffers overlay expert slots between prompts, so pp gains ~40 % at 210K *and* the resident chunk buffer shrinks (4,500 experts in VRAM instead of 4,076). Upstream now recommends `0` (the default chunk) for a discrete card; measured 2026-11-07 it resolves to the same borrowed 16K chunk on this split (same pp/tg within noise), so the explicit cap is equivalent here ([notes/deepseek/v41f1017.md](../../notes/deepseek/v41f1017.md) 20261107) |
| `DS4_RING_MB=128` | caps the prompt ring (the default sizes it from the card's link: 394 MB on the x16 monitor card); pp-neutral, +28 experts |
| `DS4_HC_STREAM=1` | keeps the hybrid-cache mix/split on the layer's own stream (HIP defaults to a side stream): +4 % tg, bit-identical math |
| `DS4_HC_MIX_SPLIT=2` | one launch for a window's hc_mix + hc_split on the side stream (exact, load-checked) |
| `DS4_ROUTE_FUSE=1` | a decoded token's router gate and the next layer's guess in one launch (exact, load-checked): the largest single tg gain of the decode fusions |
| `MAYA_DS4_SLOT_KEEP=1` | conversation slots: a conversation another request takes the model from is set aside (state + compressed KV) instead of dropped, so switching back reads only its new tokens; it persists across engine runs in `<pack>/slots` (the engine writes the RAM slots there at QUIT, so a llama-swap stop/restart reads on from them) |
| `MAYA_DS4_SLOT_CAP=8` | conversations kept in RAM (engine default 4); past it the least recently used spills to the slot folder and is read back from there |
| `MAYA_DS4_SLOT_RAM_GB=6` | steady-state RAM byte cap for the parked conversations (engine default `0` = count-limited only); a slot is tens to hundreds of MB (V4.1's compressed KV is ~1.1 KB/position per card, ~0.45 GB at 200k tokens on this pair) |

`--reserve-mib 1664` is the headroom knob, not a speed knob: the engine subtracts it per GPU before
placing experts, and the run-time floor is ~`reserve - 648 MiB - the monitor card's desktop VRAM`
(measured with the split: 598 MiB free on the monitor, 1,015 on the other card during the 85 % prefill).
Raising it costs ~12 experts a card per 100 MiB; lowering it eats the desktop's headroom first.

### GLM lanes (Maya-M and Maya-M-Derisked)

Two GLM-5.3-Flash lanes share one tuned configuration, one quant each:
`configs/glm53f-iq2s-derisked/maya-glm53f-iq2s-derisked-hip.json` (llama-swap `glm-5.3-flash-iq2s-derisked`,
Maya-M-Derisked) and `configs/glm53f-iq2s/maya-glm53f-iq2s-hip.json` (llama-swap `glm-5.3-flash-iq2s`, the vanilla
Maya-M). Each config sits in its own subfolder like every lane (see "Configs from the repo"): a lane mounts only
its folder and points `CONFIG_SRC` at it. The vanilla lane is the derisked lane's configuration copied 1:1 -
same args, env and vision values; only the shards, the lane data tree, the config folder and the config's
`model_name`/`log` differ (the quant has identical shard sizes and pack layout) - so the table below applies to
both lanes. Baseline is the setup's own config
(`maya.py --setup --backend hip --gpus 0,1 --gguf-dir /data --context 524288`), ported with the knobs below:
`--glm-pack /data/pack --max-context 524288 --expert-cache auto --prefill 16384 --kv-streaming on
--expert-profile /opt/maya/data/expert-profile-glm.bin`. Tuning record, arms and numbers:
[notes/glm/5.3-flash.md](../../notes/glm/5.3-flash.md), deployment records: [notes/maya.md](../../notes/maya.md).

| Setting | Why |
| --- | --- |
| `--max-context 524288` | 512k. KV streaming (`--kv-streaming on`, setup default) keeps 32768 positions a layer in VRAM and the latent cache in pinned RAM: 1.71 GB a card at 512k with the deployed int8 KV (3.22 GB with fp16) |
| `--kv int8` | **int8 KV** (deployed 2026-11-06): halves the latent cache to 544 B/position/layer (3.22 -> 1.71 GB a card) and the VRAM window (0.20 -> 0.11 GB), freeing 3 GiB of pinned RAM (`MemAvailable` ~10 -> ~13 GiB) and ~11 more experts. Measured pp identical and tg unchanged within +-3 %. **The slot identity does not include the KV format**, so clear `<pack>/slots` when switching (`--kv fp16` reverts). 1.1.55 defaults GLM to int8, so the flag is pinned here explicitly |
| `--prefill 16384`, `MAYA_GLM_PREFILL_MB=6144` | the prompt chunk. Auto caps a split at 8192; the engine's chunk buffers overlay expert slots between prompts, so a bigger chunk costs nothing at decode. 16K measured **+16 % pp at 40 % and +23 % at 85 %**; 32K added only ~2 % more and cost 2.7 GiB of MemAvailable plus swap. `PREFILL_MB=6144` is the loan cap that lets a 16K chunk fit (the measured borrow is 3.6-4.2 GB a card) |
| `MAYA_GLM_PRESTAGE=160` | the prompt **prestage ring** (deployed 2026-11-07). The engine skips it on a split by default (`ring_max = has_moe && n_parts_ == 1 ? kPreSlots : 0`); 160 experts of the next layer's most routed RAM-tier ones are copied into a pool-tail buffer while attention runs, when the PCIe link is otherwise idle. Costs 1,371 MB of the prompt borrow (3,381/3,994 -> 4,751/5,365 MB a card, still under the 6,144 cap) and nothing at decode: **+8.0 % pp at 40 % and +5.4-7.3 % at 85 %** (screening 973.7 / 1012.7 t/s, 2-rep final 968.5 / 994.9, controls 901.7 / 944.2); tg, floors (271 MiB other / 434 MiB monitor) and `MemAvailable` (12.8-13.1 GiB) unchanged, swap flat. 240 experts brushes CUDA1's borrow cap and measures 0.6 % slower. Mechanism: RAM-sourced demand staging 17.5 -> 0.10 GB a chunk, total PCIe 33.7 -> 17.7 GB |
| `MAYA_GLM_RESERVE_MB=1200` | headroom kept free on the **monitor** card, sizing its expert pool. The measured run-time floor is ~`reserve - 760 MiB` (engine buffers + the desktop's ~460 MiB): 437-453 MiB free at the deployed 1200 on the 1.1.55 ladder (the setup's 3072 left 2.2 GiB idle; the tuning pass's 1500 kept ~736 MiB). It stays 1200 because the picture turn is lent from the pool, not the reserve: **527 MiB idle, 1164 MiB during a 4K picture turn** (with `VISION_FOOTPRINT_MB` below; without it the encoder's shortfall dipped the turn to 28-35 MiB) |
| `MAYA_GLM_RESERVE_LATER_MB=650` | the same for the later (x4, no-display) card: `~reserve - 380 MiB` (271-274 measured at the deployed 650), i.e. ~270 MiB free. The asymmetric reserve is why the x4 card's pool can grow without over-reserving the monitor |
| `MAYA_GLM_MLA_ATTN2=h64lvc`, `MAYA_GLM_DSA_SCORE2=rl2b4` | the decode-only "take-two" kernels for the F32 MLA split attention and the sparse indexer (MLA_MMA is compiled out on HIP). **Bit-exact**; +1 % at 52K and +5-10 % tg at 200-450K, pp untouched |
| `MAYA_GLM_RAM_GB=36` | fixed pinned RAM expert tier - the DeepSeek lane's `--ssd-cache 36` analogue. Measured: 35.98 GB across the parts (17.57 + 18.41); the rest of the off-card experts stream from the SSD. 40 GB buys ~5 % tg at 85 % but costs ~5 % pp and ~3 GB of swap on this 62 GiB host |
| `MAYA_GLM_SLOTS=4`, `MAYA_GLM_SLOT_GB=16` | conversation slots on **disk** (`<pack>/slots`). GLM has no RAM-slot tier: one 512k fp16 slot is ~5.9 GB of KV, so the DS4 lane's RAM parking (`MAYA_DS4_SLOT_RAM_GB=6`) would hold a single slot. 16 GB holds ~2 full-context slots, more when shorter |
| `MAYA_GLM_SLOT_KEEP=1` | slots survive a llama-swap stop/restart (the engine writes them at QUIT; `docker stop -t 120` covers `MAYA_ENGINE_QUIT_S` 50) |
| `vision.max_tokens=12288` | the picture budget (prompt tokens) `maya-vision` resizes each image to. GLM's tower is 1 token per 28x28 px (patch 14 x merge 2 = 784 px), so a naive 4K (3840x2160) needs **10,549** tokens and 4096x2160 needs 11,244; 12288 feeds both un-downscaled. Unset, the server auto-fits 4096/2048/1024 by the encoder's measured VRAM (15% rule) - 4096 on these cards, which downscales 4K to ~2380x1344. The Qwen3.8-Flash-Next strata reference (`VISION_TOKENS=8192`) is **not** enough here: its tower is 1 token per 1024 px (patch 16), GLM's is ~30% denser |
| `VISION_FOOTPRINT_MB=4711` | the episode's other half: the encoder's VRAM lend. The server sizes it from `maya-vision --measure`, whose warm-up is a fixed 2048x2048 picture (**5,329 tokens** here), so at 12288 the measurement covers only half a 4K picture's patches and under-covers the peak by ~750 MiB - the encoder then dips into the monitor reserve (28-35 MiB free at a 4K turn). The entrypoint pins the cached footprint to **4711 MiB** (a real 4K picture's peak measured ~4.14 GB plus slack) under the key the server would use, so the engine lends ~4.6 GiB from the expert pool's tail only while a picture encodes: the 4K turn then keeps **1164 MiB** free on the monitor card and the reserve stays 1200 (the monitor pool keeps 2,265 experts, +90 over the 1950 reserve, back to the 1.1.55 level). Drop the pin when the upstream warm-up scales with the budget ([notes/TODO.md](../../notes/TODO.md)) |

Kept slots and pack rebuilds: the GLM session identity carries the session ABI (`engine=0.1.27`) **and the pack
files' sizes** (`index.txt`, `dense.bin`, `native_experts.txt`), so an engine bump alone keeps the conversations but a
pack rebuild clears them - the #861 F32-bias repack made 1.1.56 remove 4 stale conversations (4.2 GB) on the derisked
lane and 2 (227 MB) on the vanilla one. The DeepSeek identity carries the ABI plus `ds4.txt`/`engram.bin`/
`tensors.txt` and the shard sizes (no pack rebuilds in practice): its 117 conversations (7.21 GB) survived the
1.1.55 -> 1.1.56 bump and its 134 (11.51 GB), the GLM lanes' 4 + 4, survived 1.1.57 -> 1.1.58 (no pack or ABI
change). Records: [notes/maya.md](../../notes/maya.md) 20261107/20261111.

Measured on the deployed config (2026-11-11, Maya 1.1.58): the 1.1.58 A/B against 1.1.57 was unchanged within
the run-to-run spread (pp ~977 / ~997 t/s, tg ~19.1 / ~19.2 t/s at 40 / 85 %; min free 359 MiB monitor / 73 MiB
other, 11.2 GiB `MemAvailable`, swap near-full during the prompts on both versions from page-cache pressure).
With the prestage ring (2-rep final, 2026-11-07 on 1.1.56):
**pp 968.5 / 994.9 t/s, tg 19.75 / 20.65 t/s at 40 / 85 %**, `cache_n` 0 on
every rep, acceptance 0.760 / 0.788; auto split [0,24) / [24,45)+MTP, int8 KV, 16K chunks, **min free VRAM
434 MiB monitor / 271 MiB other** (targets 400 / 200), 12.8 GiB MemAvailable, swap flat, 6/6 functional gates
(chat, tool call, `reasoning_effort: none`, primes, fenced code, a picture turn). The 4K
picture turn (2026-11-06, camera check; the vision budget unchanged): `prompt_tokens` 10,579 = the 137x77 grid
with no downscale, answer correct, **527 MiB monitor free idle, 1164 MiB during the turn** (349 MiB on the other
card), 2,265 + 2,304 = 4,569 experts in VRAM; the encoder lends 4.76 GB from the pool for the ~15 s encode and
gives it back (the 1.1.58 gate run: 10,577 tokens, no downscale). The tuning arms, the 1.1.53
numbers and the against-setup deltas: [notes/glm/5.3-flash.md](../../notes/glm/5.3-flash.md); the 1.1.57 and
1.1.58 A/Bs: [notes/maya.md](../../notes/maya.md) 20261110/20261111; the #861 repack and the picture budget:
[notes/maya.md](../../notes/maya.md) 20261107. Ready in ~31 s cold, stop 1-2 s.

The vanilla lane (2026-11-06 bring-up, same config) matches it line for line: auto split `[0,24)` /
`[24,45)+MTP`, 2,265 + 2,304 = 4,569 experts, int8 KV, 16K chunks, 35.98 GB RAM tier; ready in 28 s; 7/7
functional gates including the 4K picture (`prompt_tokens` 10,579, no downscale); min free VRAM 445 / 286 MiB,
13.5 GiB `MemAvailable`, swap flat. Its 1.1.58 restart repeats 7/7 gates and 10,577 picture tokens. Record:
[notes/maya.md](../../notes/maya.md) 20261106/20261107/20261110/20261111.

The kept knobs above are what moves this lane; the rejected ones (per-card CPU lanes, promotion limits,
`SPEC_WIN`, `MV_RDNA`, `GU22_TUNED`, DFlash2, no MTP at all (`MAYA_GLM_NO_MTP`, -19 % tg), the disk-skip
approximation (`DISK_SKIP_W`/`_FROM`, a residency-dependent quality trade, not adopted), a manual split, a
40 GB RAM tier, `MAYA_GLM_P2P`, the decode lookahead (`MAYA_GLM_AHEAD`, -9 to -20 % tg in a dose-dependent
sweep), `MAYA_GLM_PREFETCH_N`, the prompt CPU gate (`MAYA_GLM_PREFILL_CPU_GATE=0`, neutral), a traffic-traced
expert profile (-2.7 % vs the shipped one) and 32K chunks on the prestage base (-2 to -3 % pp)) - and, one day
later, the 2026-11-09 CPU-vs-PCIe share pin + 4-piece disk reads (tg52/short-decode wins that reversed to
-4 to -14 % tg on the 2 %-out production ladder; `READ_CHUNKS=4` alone: +2.9 % at 40 %, -7.4 % at 85 %); the
reasons are in [notes/glm/5.3-flash.md](../../notes/glm/5.3-flash.md) 20261109. The layer split is auto on
purpose: moving layers
to the monitor card buys up to +11.6 % pp but costs 6-13 % tg (the pipelined decode costs the slower
part), and tg is this lane's weak axis.

Because that copy resets whatever changed the live config — dashboard edits, and maya.py's own rewrites after an
upstream bump (schema migrations) — the entrypoint first **summarizes the difference** in the log, e.g.

```
maya: /data/config/maya-deepseek-2bpw-hip.json differs from the repo copy (args (- --ssd-cache 36 + --ssd-cache off); only in live: expert_profile). The repo copy replaces it: dashboard edits and maya's own rewrites are reset - port wanted changes into the repo file.
```

That line is how a maya-side config change is noticed at all: when it fires on a lane that nobody touched, port
the named changes into `deploy/maya/configs/<lane>/maya-<quant>-hip.json`, commit, and `git pull` on the runner.
- The macro also bind-mounts `deploy/maya/entrypoint.sh` over the image's own (the same file, mode 755 in git), so
  an entrypoint change ships with a git pull + lane restart instead of a ~30 GB image rebuild — the maya analogue
  of the strata fork mount. The DeepSeek lane sets `DS4_VISION_TOKENS=1140`: the picture budget the entrypoint
  writes into the pack's `ds4.txt` at every start (the strata fork's change 6 mirrored; 1140 is the tower's
  10,240-patch maximum, where Maya's pack default 1024 downscales a 4K screenshot to 1680×966; notes/maya.md).

## Refusal ablation (`:ablation` alias)

Maya 1.1.59 can project a refusal direction out of every residual write for the requests that ask (upstream
`docs/SETTINGS.md`, "Refusal ablation per request"). Each maya lane loads its model's direction file at start and
leaves it off; a request turns it on per call, with no restart:

- **Plain requests are untouched**: the server injects `--ablate FILE --ablate-default off` from the config's
  `"ablate"` key (the file must exist at start, else the switch stays hidden), and a request without the field
  never carries the projection - its engine line and its answer are the same as without a direction file.
- **`<model-id>:ablation`** (per-lane `filters.setParamsByID` in `deploy/llama-swap.config.yaml`, the `:npt`
  pattern) adds `"ablate": true` to any JSON body - OpenAI, Anthropic and Maya's model API all read it, and the
  maya server never validates the body's `model` field. The engine line then carries `GEN ablate=1`
  (`INFO ablate=1 ablate_default=0` at load).
- The dashboard's Models-page switch (`ablate.default`) stays off; the alias is the only activation path.

| Lane | Direction file (in the lane tree's `refusal-directions/`, = `/data/refusal-directions/` in-container) | sha256 (head) |
| --- | --- | --- |
| `dsv41f1017-iq2xxs` | `DeepSeek-V4.1-Flash-refusal-direction.gguf` (21,024 B; layer 25, dim 5120, scale 1.0) | `09ab0ac8…` |
| `glm53f-iq2s` | `GLM-5.3-Flash-refusal-direction.gguf` (16,928 B; layer 31, dim 4096, scale 1.0) | `ecf89232…` |

The **`glm53f-iq2s-derisked` lane carries no ablation on purpose**: the Maya-M-Derisked quant is already an
abliterated transplant, and projecting a refusal direction out of it would ablate it twice.

Both files are control-vector GGUFs from the models' Maya repos (`peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF`,
`peasantsmith/GLM-5.3-Flash-Maya-GGUF`); the GLM one is from the same repo the GLM shards come from (fitted on
Maya-S24, the lanes pack Maya-M / Maya-M-Derisked). The DeepSeek one was fitted on Maya's Maya-S quant while this
lane packs ISTA-DASLab IQ2_XXS - same base model and hidden size, and the transfer is checked with
`tools/ablate_fit.py eval` before the alias is trusted (refusal rate on held-out prompts with ablation off and on,
plus coherence; notes/deepseek/v41f1017.md).

Operational notes:

- The projection covers every residual write of the main layers on every path - prompt chunks, verify windows,
  batched `--parallel` rows, both cards of a split; maya-ds4 copies the direction to every device and projects at
  each `hc_post`, so the `--tp 2` lane is covered (only MiniMax refuses tensor parallel). The draft heads only
  propose; the main layers verify, so the kept tokens are the ablated model's.
- Switching ablate on a live conversation restarts that conversation's cache (no cached prefix, restore point,
  kept conversation or shared start crosses the two settings), and an ablated conversation is never kept as a
  slot: a `:ablation` request on a conversation the plain model already holds pays a full prompt re-read.
- An upstream refit replaces the file: re-run the one curl per lane (the pinned URLs and sha256s are in
  notes/maya.md); the lane picks it up at its next restart, no image rebuild.

## Build it yourself

On a machine with Docker and ~40 GB free (the ROCm SDK, the compile and the image), e.g. fedora-tuf:

```sh
git clone --depth 1 --branch v1.1.59 https://github.com/mw00/project-maya.git maya   # next to this repo
docker build -f deploy/maya/Dockerfile \
  --build-arg BASE_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.2.1-complete \
  --build-arg ROCM_VERSION=7.2.1 \
  --build-arg LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d \
  --build-arg HIP_ARCH=gfx1100 \
  --build-arg ENGINES=maya,maya-ds4 \
  --build-arg MAYA_REV=v1.1.59 \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t llm-ops-maya:local .
```

The compile runs with at most 4 jobs (`maya.py`'s own HIP heuristic: half the cores, 2-4 GB per clang). No GPU is
needed to build; the workflow's smoke tests (engine `--help`/`--version`, `ldd`, the stamp check,
`maya.py --help`) run without one. `maya.py --check` needs the GPU: `docker run … IMAGE --check --backend hip --gpu 0`.

After an upstream bump, move `MAYA_REF`/`LLAMA_COMMIT`/the table version in the workflow's `env:` as one reviewed
change, and re-run the workflow. The tag is manual-only.
