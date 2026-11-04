# Maya (AMD / HIP) deployment

The Maya image is **our own build** (pushed by the workflow, tag `1.1.53-rocm-gfx1100`), made by
[`.github/workflows/maya-image.yml`](../../.github/workflows/maya-image.yml). It carries Project Maya `v1.1.53`
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

Current image: `ghcr.io/hurui200320/llm-ops-maya:1.1.53-rocm-gfx1100` (Maya `v1.1.53`, GLM + DeepSeek engines,
`gfx1100`). The workflow `env:` block pins the build inputs; the image digest of each build is in
[notes/maya.md](../../notes/maya.md).

| Input | Pin |
| --- | --- |
| Maya source | `mw00/project-maya` tag `v1.1.53` |
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
  ghcr.io/hurui200320/llm-ops-maya:1.1.53-rocm-gfx1100
```

`1.1.53-rocm-gfx1100` is the image's only tag; `ghcr.io/hurui200320/llm-ops-maya:latest` is not published.

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
  ghcr.io/hurui200320/llm-ops-maya:1.1.53-rocm-gfx1100 \
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
docker run -d --name maya ... ghcr.io/hurui200320/llm-ops-maya:1.1.53-rocm-gfx1100
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

```
/data/                                   one tree per model (the strata pattern)
  model/                                 READ-ONLY mount of the production shard folder
                                         (the GGUFs the engine reads in place; an empty dir on the host -
                                         it only has content inside the container)
  vision/                                the vision tower, built from the mount's mmproj (vision.bin, vision.json)
  pack/maya-ds4/                         the pack: ds4.txt (geometry + model_dir /data/model), tensors.txt,
                                         engram.bin, tokenizer/, the engine's expert profile and caches
  config/maya-<quant>-hip.json           the installed configs the entrypoint serves from (newest wins)
  models/                                only SETUP-installed (Maya-managed) models land here
```

Per-start settings (context, expert cache…) live in the config; the dashboard edits it. Logs go to
`/opt/maya/maya-<quant>-hip.log` in the container layer (mount nothing there; use `docker logs` for the server).

### Configs from the repo (lanes)

`deploy/maya/configs/` holds version-controlled configs for lanes (one file per lane, named `maya-*-hip.json` so
the entrypoint's glob finds them). Mount the folder read-only and point `CONFIG_SRC` at it; the entrypoint copies
the JSONs into `/data/config` at **every** start, so the repo file is the declared state (the equivalent of the
strata fork's `STRATA_SET` env) and dashboard edits last until the next lane restart. Never mount a repo config
writable straight into `/data/config`: maya.py touches (and may rewrite) the config on every start, which would
dirty the checkout. The DeepSeek lane's config is `configs/maya-deepseek-2bpw-hip.json` (512k, `--ssd-cache 36
--reserve-mib 1024 --suffix-draft 0`, env `DS4_INDEX_TC=1`; rationale and numbers: notes/maya.md).

Because that copy resets whatever changed the live config — dashboard edits, and maya.py's own rewrites after an
upstream bump (schema migrations) — the entrypoint first **summarizes the difference** in the log, e.g.

```
maya: /data/config/maya-deepseek-2bpw-hip.json differs from the repo copy (args (- --ssd-cache 36 + --ssd-cache off); only in live: expert_profile). The repo copy replaces it: dashboard edits and maya's own rewrites are reset - port wanted changes into the repo file.
```

That line is how a maya-side config change is noticed at all: when it fires on a lane that nobody touched, port
the named changes into `deploy/maya/configs/<lane>.json`, commit, and `git pull` on the runner.
- The macro also bind-mounts `deploy/maya/entrypoint.sh` over the image's own (the same file, mode 755 in git), so
  an entrypoint change ships with a git pull + lane restart instead of a ~30 GB image rebuild — the maya analogue
  of the strata fork mount. The DeepSeek lane sets `DS4_VISION_TOKENS=1140`: the picture budget the entrypoint
  writes into the pack's `ds4.txt` at every start (the strata fork's change 6 mirrored; 1140 is the tower's
  10,240-patch maximum, where Maya's pack default 1024 downscales a 4K screenshot to 1680×966; notes/maya.md).

## Build it yourself

On a machine with Docker and ~40 GB free (the ROCm SDK, the compile and the image), e.g. fedora-tuf:

```sh
git clone --depth 1 --branch v1.1.53 https://github.com/mw00/project-maya.git maya   # next to this repo
docker build -f deploy/maya/Dockerfile \
  --build-arg BASE_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.2.1-complete \
  --build-arg ROCM_VERSION=7.2.1 \
  --build-arg LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d \
  --build-arg HIP_ARCH=gfx1100 \
  --build-arg ENGINES=maya,maya-ds4 \
  --build-arg MAYA_REV=v1.1.53 \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t llm-ops-maya:local .
```

The compile runs with at most 4 jobs (`maya.py`'s own HIP heuristic: half the cores, 2-4 GB per clang). No GPU is
needed to build; the workflow's smoke tests (engine `--help`/`--version`, `ldd`, the stamp check,
`maya.py --help`) run without one. `maya.py --check` needs the GPU: `docker run … IMAGE --check --backend hip --gpu 0`.

After an upstream bump, move `MAYA_REF`/`LLAMA_COMMIT`/the table version in the workflow's `env:` as one reviewed
change, and re-run the workflow. The tag is manual-only.
