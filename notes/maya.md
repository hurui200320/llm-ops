# 20261104 Project Maya on gfx1100: image, DeepSeek setup, and the strata -> maya switch

Project Maya (upstream [`mw00/project-maya`](https://github.com/mw00/project-maya)) is a Strata sibling: a Python
installer/dashboard (`maya.py`) that compiles its own C++ engines (`maya` for GLM, `maya-ds4` for DeepSeek) and serves
its own quants of GLM-5.3-Flash (321 B), DeepSeek-V4.1-Flash-1017 (748 B), and MiniMax-M3.1-Flash (428 B) over
OpenAI- and Anthropic-compatible APIs with vision and hotness-profiled expert placement across VRAM, pinned RAM, and
NVMe. It targets our 2× RX 7900 XTX pair (`gfx1100`, 24 GB each; GPU1 behind the Z790 PCH at x4; 62 GiB host RAM) via
its HIP backend (`docs/AMD_MAYA.md` upstream).

Upstream ships no Docker support (`maya.py` compiles on the user's host at first start). Our image compiles the
binaries once and writes the rebuild stamps `maya.py` checks, so first start only provisions the model pack and serves.
The DeepSeek lane moved to Maya the same day because it beat the tuned `strata-dsv41` lane on every axis, and the
Strata DeepSeek build was retired.

## Pins (`v1.1.52`)

| What | Value |
| --- | --- |
| Maya | tag `v1.1.52` (`mw00/project-maya`), commit `3c935e411cd6625a9af0f292dd0af52598e866d8` |
| llama.cpp | `3cf03257f219afbe7334045ff7c6a06ac68c627d` (gated against `setup.py` by the workflow) |
| ROCm base | `docker.io/rocm/dev-ubuntu-24.04:7.2.1-complete` |
| Arch / hipBLASLt | `gfx1100` only; `tools/hip/gfx1100-glm-hipblaslt-100202.txt` (shipped alongside `100401`, `100500`) |

## Setup: the image

### Installer requirements and ROCm base

`check_hip_pc()` runs at **every start** (not only setup) and requires `hipcc` ≥ 7.0 at `<root>/bin/hipcc`, ROCm's
clang (`<root>/llvm/bin`, `<root>/lib/llvm/bin`, a versioned slot, or `<root>/bin`), `libhipblas.so*` under
`<root>/lib|lib64`, a `c++` on `PATH`, AVX2, and at least one supported KFD GPU (`<root>` resolves `ROCM_PATH` →
`/opt/rocm` → newest `/opt/rocm-[0-9]*` → `/usr`). A libs-only runtime image cannot start Maya; the full SDK must ship.

Unlike Strata (which bundles its ROCm libs in upstream's prebuilt tarball), Maya links system ROCm and re-checks it at
every start. TheRock wheels (`7.10.0a20251120`) were rejected because they do not match Maya's ROCm 7.2.1 validation
line or its `100202` gfx1100 hipBLASLt table and would need a hand-assembled SDK. Instead, both build and runtime
stages use AMD's `rocm/dev-ubuntu-24.04:7.2.1-complete` (~7.4 GB compressed; Clang 22, hipBLASLt 1.2.2 = `100202`).

### Build, stamps, layout, and entrypoint

Engine build (`compile_engine_hip`, Linux branch), reproduced in `deploy/maya/Dockerfile` with `max(1, min(4, cores/2))`
jobs against `setup.py`'s pinned `third_party/llama.cpp`:

```
-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF
-DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF
-DSTRATA_PORTABLE=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
-DCMAKE_HIP_COMPILER=<rocm clang++> -DCMAKE_PREFIX_PATH=/opt/rocm -DSTRATA_GGML_DIR=<llama.cpp>
```

Vision (`compile_vision`, Linux HIP branch) runs `tools/vision/glm5next_patch.py` (idempotent) on llama.cpp, then
builds target `maya-vision` → `build-vision/bin/maya-vision` from `tools/vision` with `-DSTRATA_VISION_HIP=ON
-DSTRATA_VISION_HIP_TARGETS=gfx1100 -DAMDGPU_TARGETS=gfx1100 -DCMAKE_HIP_ARCHITECTURES=gfx1100
-DCMAKE_EXE_LINKER_FLAGS=-no-pie -DSTRATA_PORTABLE=ON`.

`deploy/maya/stamps.py` writes `build-hip/MAYA-BUILD.json` (`{backend, src, archs, rocm, gpu_ids, llama, lib_dirs,
date, targets}`) and `build-vision/MAYA-BUILD.json` using Maya's own `source_hash` and the **resolved** ROCm root
(`/opt/rocm-7.2.1`), which `build_step`/`vision_step` require to skip recompilation. `STRATA_HIPBLASLT_TUNING` lists
every `tools/hip/<arch>-glm-hipblaslt-*.txt` for the card's arch; the engine selects the table matching its runtime
hipBLASLt version (`100202`).

- **Build & runtime stages**: Build stage compiles `maya`, `maya-ds4`, and `maya-vision`, writes `BUILD_ID` (so
  `--version` reports the git commit from source archives) and stamps, and populates `.venv` (`setup.PY_PACKAGES` +
  vendored `textual` wheels + `.strata-pip.json`). Runtime stage (`rocm/dev-ubuntu-24.04:7.2.1-complete` + `libgomp1
  curl python3-venv build-essential`) copies `/opt/maya` without CMake object trees.
- **`/entrypoint.sh`**: With no args, serves the newest `/data/config/maya-*-hip.json` (`--backend hip --config …
  --host 0.0.0.0 --port $PORT`); any args pass straight to `maya.py`.
  - `SETUP=1`: runs first-time guided setup and copies the generated config to `/data/config/`.
  - `PACK_BUILD=1`: self-provisions `/data/vision` and `/data/pack/maya-ds4` from a read-only `/data/model` GGUF mount
    when missing, and normalizes `model_dir` to `/data/model`.
  - `CONFIG_SRC`: directory of repo-declared config JSONs copied into `/data/config` at every start (Maya has no
    `STRATA_SET` equivalent, and a writable file bind would be dirtied by `maya.py`'s `touch()`/migrations). Logs a
    diff summary before overwriting live edits.
  - `DS4_VISION_TOKENS` / `DS4_VISION_MIN_PIXELS`: atomically rewrites `vision.max_image_tokens` (clamped at 1140) in
    every `maya-ds4` pack's `ds4.txt` at start, surviving `pack_gguf.py` repacks on upstream bumps.

### Build verification and the ggml ISA fix

| What | Value |
| --- | --- |
| Local test image | `llm-ops-maya:test`, 30,118,810,113 B uncompressed (`sha256:58e70296b93bd0652ece5f6a5d025e6780cc0c820b951d5c5eaf385ab843e251`); ~0.65 GB over ROCm base (venv 229 MB, `build-hip` 87 MB, `build-vision` 80 MB, llama.cpp 173 MB) |
| Build time (24c, 4 jobs) | `maya` + `maya-ds4` 173 s, `maya-vision` 195 s, venv 31 s (~8 min after base pull) |
| Binaries & stamps | `maya` 42,719,832 B, `maya-ds4` 48,180,928 B (`src=d9009cf853f7c0fa`), `maya-vision` 83,544,176 B (`src=4bc493dc7f779c19`), `rocm=/opt/rocm-7.2.1`, hipBLASLt `1.2.2` (`100202`) |

**The ggml ISA bug:** the first CI-built image (`sha256:e2c3c53b5a7b…`) died with **exit 132 (SIGILL)** in
`ggml_cpu_init` at first `Model::load` on the AVX2-only i7-13700K host because `maya-ds4` statically links llama.cpp's
`ggml-cpu`, which baked in the GitHub Actions Xeon runner's AVX-512 ISA. Fixed in `deploy/maya/Dockerfile` by passing
`-DSTRATA_PORTABLE=ON` (`GGML_NATIVE=OFF` + explicit `GGML_AVX/AVX2/FMA/F16C/BMI2=ON`) to both CMake builds and
asserting `GGML_NATIVE:BOOL=OFF` and `GGML_AVX2:BOOL=ON` in both `CMakeCache.txt` files at build time.

## Setup: the DeepSeek pack and data tree

Maya's guided setup only recognizes its own manifest quants, but its standalone pack CLIs accept the ISTA-DASLab
6-shard 2.00 bpw set as-is:
- `tools/ds4/pack_gguf.py` handles llama.cpp's `deepseek41` layout (`ple.*` Engram metadata, shard 6 `split_ple`,
  `engram_kv`/`engram_gate`, `exp_probs_b_vl`, 2-D `attn_output_a`, and the #500 missing-DSpark-keys workaround). The
  engine natively reads the ISTA dtype mix (IQ1_M / IQ2_XXS / IQ2_XS experts, Q5_K / Q8_0 attn, BF16 routers, F32
  norms/sinks/HC) and exposes `/tokenize` (saving ~18 min of calibration prefills per `run_longctx.py` ladder).
- `tools/ds4/vision_pack.py --mmproj` builds `/data/vision/vision.bin` (~0.97 GB, 266 tensors verified) from unsloth's
  `mmproj-BF16.gguf` (`9a1ef7c4…`).

```
/mnt/unenc-xfs/maya/dsv41f1017-iq2xxs/          lane data tree (mounted at /data)
  model/           READ-ONLY mount of /mnt/unenc-xfs/models/maya/ISTA-DASLab-dsv41f1017 (production shards)
  vision/          vision.bin + vision.json
  pack/maya-ds4/   ds4.txt (model_dir /data/model), tensors.txt, engram.bin (517,980 B), tokenizer/, caches
  config/          live copy of deploy/maya/configs/... (via CONFIG_SRC)
```

Why `/data/model` mounts inside writable `/data`: `pack_gguf.py` records the vision tower in `tensors.txt` as
`../vision/vision.bin` **relative to `model_dir`**, so `model_dir`'s parent must be the writable data tree. On first
start (`PACK_BUILD=1`), the entrypoint builds `vision.bin` and packs all `.gguf` files except `mmproj-*.gguf` (*"47550
tensors in 7 files … Engram 2 layers x 24 hash columns, the vision tower"*). Setting `DS4_VISION_TOKENS=1140` raises
the picture cap from 1024 (1680×966 on 4K) to the tower's 10,240-patch softmax limit (1764×1008 on 4K).

## Testing: DeepSeek-V4.1-Flash 2.00 bpw (`v1.1.52`)

- **Functional gates (6/6)**: `SMOKE-OK` chat; `get_weather` → `{"city":"Paris"}`; `reasoning_effort: none` → `42`;
  first five primes; `def sum_list` code block; 420×420 and 3840×2160 red/blue pictures (~1,127 image tokens on 4K in
  14–16 s). Cold load 19–25 s (vs Strata's ~41 s); `docker stop` 0.9 s with VRAM freed.
- **Speed A/B (512k, `--ssd-cache 36 --reserve-mib 900`, auto `[0,20]`/`[21,39]` split, `run_longctx.py --ctx 524288
  --in-pcts 40,85 --out-pct 1 --reps 1`, `cache_n 0`)**:

  | Arm | pp 40 % | tg 40 % | pp 85 % | tg 85 % | Notes |
  | --- | ---: | ---: | ---: | ---: | --- |
  | Strata 0.1.56 (deployed) | 637.5 | 19.5 | 568.9 | 18.2 | with `STRATA_DS4_HIPK=1 DS4_FAST_ATTN=1` |
  | Maya A0 (`--suffix-draft 3` default) | 656.9 | 23.5 | 628.6 | 23.4 | suffix drafter accepted 41%/13% but slowed decode |
  | Maya A1 (`--suffix-draft 0`) | 696.4 | 24.0 | 621.6 | 24.7 | +20 MiB/card freed (4,231 experts in VRAM) |
  | **Maya A2 (`+ DS4_INDEX_TC=1`, kept)** | **725.1** | **24.8** | **704.7** | **25.4** | **+13.7% / +27% at 40%, +23.9% / +40% at 85% vs Strata** |

- **52K War-and-Peace knob probes on A1 (3 cold reps each)**:
  - P0 base: 632.6 pp / 21.9 tg.
  - P1 `DS4_INDEX_TC=1` (WMMA indexer scores, #801): 646.2 pp (+2.0–2.4% at 52K, +4.1%/+13.4% at 40%/85% of 512k),
    22.2 tg — **kept**.
  - P2 `STRATA_DS4_PREFILL_FUSED=1`: 640.3 pp (+1.2%, noise), 21.8 tg — **rejected**.
  - P3 `--ssd-cache off`: 629.5 pp, 20.6 tg (−6% tg, frees 33 GiB host RAM: `MemAvailable` 18 → 51 GiB) — **rejected**.
- **Placement (A2)** — note Maya's `CUDA0` is the x16 monitor card (opposite of sysfs): CUDA0 `[0,20]` weights 2.87 GB,
  experts 17.79 GB (2,164/8,064), caches 0.48 GB, 886 MiB free; CUDA1 `[21,39]`+head weights 2.83 GB, experts 18.74 GB
  (2,067/7,296), caches 0.19 GB, 882 MiB free; **4,231 experts in VRAM**, 11,133 on CPU (95.9 GB in place, 16 threads,
  47.7 GB page-cache read-ahead), RAM store 3,512 slots (36.0 GB, filled in 4.7 s). Kept initial config (`--reserve-mib
  1024` in `deploy/maya/configs/`): `--pack /data/pack/maya-ds4 --max-context 524288 --tp 2 --ssd-cache 36
  --reserve-mib 1024 --suffix-draft 0`, env `DS4_INDEX_TC=1`.

## Deployment: the lane moves to Maya; strata retires

`deploy/llama-swap.config.yaml` now serves **`dsv41f1017-iq2xxs`** on `ghcr.io/hurui200320/llm-ops-maya:1.1.52-rocm-gfx1100`
via `docker-server-maya-rocm`, bind-mounting `deploy/maya/entrypoint.sh` and `deploy/maya/configs/` so entrypoint and
config changes ship with a `git pull` + restart instead of a ~30 GB image rebuild. End-to-end provisioning and negative
paths (missing shard mount, unset `PACK_BUILD`, live config diff) were verified on a fresh data tree. The Strata
`strata-dsv41` build and data tree were removed the same day ([notes/strata.md](strata.md) 20261104;
[notes/deepseek/v41f1017.md](deepseek/v41f1017.md) 20261104).

# 20261104 Maya 1.1.53: a parity bump

Mechanical parity bump to `v1.1.53` (commit `7c4aa56ed7faea2788705d8e85dd58e892d19f4b`, 65 commits over `v1.1.52`;
llama.cpp pin, ROCm 7.2.1 base, `gfx1100` arch, and `100202` hipBLASLt table unchanged). Workflow run `41844250080`
(30m13s) pushed `ghcr.io/hurui200320/llm-ops-maya:1.1.53-rocm-gfx1100`, digest
`sha256:c5794fe98f8135ffe9e6e4811269e0d8a97c09265a3d62aafc39ec1ec7505604` (base
`rocm/dev-ubuntu-24.04@sha256:3db551c4e1229aac1857ac44fcb6141bb749f41348eb572452f11279153c13c3`).

- **Upstream changes**: Build inputs and pack tools (`pack_gguf.py`, `vision_pack.py`) are unchanged. `maya-ds4` adds
  `src/ds4/ds4_lane_parity.cpp`, BF16 head / shard layout checks (#806), GPU-fault engine restart on next request
  (#688), and opt-in knobs (all off by default: NVIDIA/gfx950 `MAYA_DS4_Q8MV_TC` / `EXL3_WMMA_GFX950`, prompt-path
  `DS4_FLOW_COPY_DMA` / `DS4_AI_RUN` / `DS4_CHUNK_AUTO` / `DS4_YOCO_FLOW` / `DS4_LENT_RAM_GB` / `MAYA_PROMPT_MEMO`,
  `MAYA_DS4_LANE_PARITY`, CPU EXL3 switches).
- **Verification**: Tested both direct and via llama-swap on `/mnt/unenc-xfs/maya/dsv41f1017-iq2xxs` (`maya-ds4 1.1.53
  (build 7c4aa56ed7faea2788705d8e85dd58e892d19f4b)`, ready in ~17 s, vision 1140, 512k ctx): 6/6 functional gates
  passed.

# 20261105 DeepSeek lane tuning, conversation slots, and per-lane config folders

- **Speed & split tuning (`maya-ds4` 1.1.53)**: Replaced the carried-over Strata settings with Maya-tuned knobs in
  `deploy/maya/configs/dsv41f1017-iq2xxs/maya-deepseek-2bpw-hip.json`: `--reserve-mib 1664` (sizing for `min free ~=
  reserve - 648 MiB - desktop` ≥ 500 MiB on the monitor card), `DS4_CHUNK_GFX11=16384` (borrowed 16K prefill chunk),
  `DS4_RING_MB=128`, decode launch fusions (`DS4_HC_STREAM=1 DS4_HC_MIX_SPLIT=2 DS4_ROUTE_FUSE=1`), and manual layer
  split `DS4_SPLIT=18,22` (moving layers off the desktop-bearing monitor card CUDA0 `[0,17]` / CUDA1 `[18,39]`+head).
  512k 2-rep ladder reached **1,323.25 pp / 26.65 tg at 40% and 1,169.20 pp / 27.10 tg at 85%**, 4,559 experts in
  VRAM, min free 598 / 1,015 MiB. Full arms, rejected knobs, and split curve:
  [notes/deepseek/v41f1017.md](deepseek/v41f1017.md) 20261104 and 20261105.
- **Conversation slots (`MAYA_DS4_SLOT_KEEP=1 MAYA_DS4_SLOT_CAP=8 MAYA_DS4_SLOT_RAM_GB=6`)**: Unlike Strata's dsv41
  server, Maya's server sets no `SLOT_*` defaults. Slots hold compressed KV in RAM/disk (never VRAM). The 6 GB RAM cap
  holds up to 8 slots alongside the 36 GB expert store on our 62 GiB host; on `QUIT` (`MAYA_ENGINE_QUIT_S` 50 s, covered
  by `docker stop -t 120`), RAM slots flush to `/data/pack/maya-ds4/slots` (16 GB disk cap) to survive llama-swap stops.
  `MAYA_DS4_KEEP_THINKING` is left unset (tool-carrying requests already retain prior reasoning).
- **Per-lane config folders**: Moved the DeepSeek config into `deploy/maya/configs/dsv41f1017-iq2xxs/` (matching
  `glm53f-iq2s-derisked/`) and removed the shared `CONFIG_SRC` default from `docker-server-maya-rocm`. Because the
  entrypoint serves the newest copied JSON in `/data/config`, each lane must mount only its own subfolder
  (`/repo-configs-ds4`, `/repo-configs-glm`, `/repo-configs-glm-vanilla`).

# 20261105 GLM-5.3-Flash Maya-M-Derisked lane bring-up and tuning

First lane on the image's GLM engine (`build-hip/maya`): `peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF` (3
shards, 47.7 + 47.8 + 20.5 = 116.0 GB; IQ2_S gate/up, IQ3_XXS down, Q6_K attention, NextN MTP `blk.45` inside the
shards).

## Layout (per-file read-only binds) and one-shot setup

Unlike `maya-ds4` (which reads `model_dir` from `ds4.txt`), `src/core/glm_model.cu` opens every shard at
`<pack>/../<shard>.gguf` and setup requires `--gguf-dir` to be writable, while `/entrypoint.sh` reserves `/data/model`
for DS4. Each production shard is therefore bound read-only **per file** at `/data/<shard>.gguf` beside writable
`/data/pack` (Docker leaves 0-byte root-owned mount-point files in the host data tree, which `glm_files()` ignores):

```
docker run ... -v /mnt/unenc-xfs/maya/glm53f-iq2s-derisked:/data \
  -v <prod>/GLM-…-0000X-of-00003.gguf:/data/GLM-…-0000X-of-00003.gguf:ro (×3) \
  IMAGE --setup --backend hip --gpus 0,1 --yes --no-start --gguf-dir /data --context 524288 --host 0.0.0.0 --port 8080
```

`--gpus 0,1` is required (without it `check_hip_pc` writes a single-GPU `"gpu": [0]` config). Setup verified all three
shard SHA-256s (~48 s per 47.7 GB shard), built `/data/pack` (`iq_pack.py --compat-bf16`: 1,257 tensors, 317 native,
arena/dense 0.88 GiB, vocab 154,880, merges 321,649; `blk.45` read directly from shards as 173 MB dense + Q3_K/Q4_K
experts on CUDA1), and fetched `/data/vision` (mmproj 1,075 MB + vocab 9.2 MB).

## Initial verification and 2026-11-05/06 tuning summary

- **Initial bring-up** (`--max-context 524288`, fp16 KV with `--kv-streaming on` 32,768 pos/layer in VRAM + 3.22 GB
  pinned/card, `MAYA_GLM_RAM_GB=36`, `MAYA_GLM_RESERVE_MB=3072`, disk slots `MAYA_GLM_SLOTS=4 MAYA_GLM_SLOT_GB=16
  MAYA_GLM_SLOT_KEEP=1`): cold load ~33 s / warm ~21 s, auto split `[0,24)` / `[24,45)`+head+MTP, 4,034 experts in
  VRAM (16.40 + 16.50 GB), 35.98 GB RAM tier, 8,192 prefill chunk, vision encoder measured at 2.63 GB (2.72 GB lent).
  Passed 6/6 functional gates and verified disk-slot persistence across container restart (`cache_n 1730` of 1,744
  tokens restored from a 176.6 MB slot).
- **2026-11-05/06 tuning pass ([notes/glm/5.3-flash.md](glm/5.3-flash.md))**: Kept per-card reserves
  (`MAYA_GLM_RESERVE_MB=1500` → `1200`, `MAYA_GLM_RESERVE_LATER_MB=700` → `650`), `--prefill 16384` +
  `MAYA_GLM_PREFILL_MB=6144`, bit-exact decode kernels `MAYA_GLM_MLA_ATTN2=h64lvc` / `MAYA_GLM_DSA_SCORE2=rl2b4`, and
  `--kv int8` (freeing ~3 GiB host RAM at neutral speed), lifting the 512k ladder from 748/740 pp and 16.9/15.8 tg to
  **897/933 pp and 18.3/19.3 tg** at 40/85% with 4,517–4,527 experts in VRAM.

# 20261106 Maya 1.1.55 upgrade and 1.1.53 A/B

Upgraded to `v1.1.55` (commit `c4effee67cb95fb4303720919b075fddf04b4e69`, 52 commits over `v1.1.53` across same-day
`v1.1.54` + `v1.1.55`; llama.cpp `3cf03257…`, ROCm `7.2.1-complete` base `sha256:3db551c4…`, `gfx1100`, and `100202`
hipBLASLt table unchanged). Workflow run `42174794939` (30m) pushed
`ghcr.io/hurui200320/llm-ops-maya:1.1.55-rocm-gfx1100`, digest
`sha256:04ac792dac37ebe7768df1a433e4bc3bf644ad149bc3ac09a69cd7b86c759463` (30,126,717,744 B).

## Upstream changes reaching our image and lanes

- **Build & pack tools**: `setup.py`, Dockerfile, `stamps.py`, `entrypoint.sh`, and `pack_gguf.py`/`vision_pack.py` are
  unchanged (`iq_pack.py` gained a `qwen4exp` branch only; CMake added `q5_0` MMQ instances). `check_hip_pc` relaxed
  ROCm-6 and non-AVX2 checks upstream, while our image keeps `STRATA_PORTABLE=ON` (AVX2 baseline) asserted.
- **New engine defaults (`0` disables each)**: `DS4_AI_RUN=1` (prompt-sized indexer slabs), `MAYA_DS4_SPEC_HEAD=1`
  (first ring of experts pre-copied from pinned RAM before routing), `MAYA_DS4_DRAFT_SPREAD=1`,
  `MAYA_DS4_LOAD_INTERLEAVE=1`, server `MAYA_PROMPT_MEMO=1`, engine `--decode-cache-every 1024`, `DS4_INDEX_TC=1` now
  default on gfx11, and GLM latent KV now defaults to `int8` (already pinned in our GLM config). All kept DS4 and GLM
  knobs remain valid, and **slots survive the bump** (engine version is no longer part of slot identity: DS4 reloaded
  13 conversations / 2.69 GB, GLM 1 / 1.78 GB). Upstream also added OpenAI Responses API support for Codex CLI (#726).

## Verification and interleaved 512k A/B (1.1.53 vs 1.1.55)

Both lanes passed 6/6 functional gates through llama-swap (`dsv41f1017-iq2xxs` at 4,537 VRAM experts vs 4,557 on
1.1.53 due to 128-row `DS4_AI_RUN` + spec-head buffers). Interleaved direct-container 512k ladders (A, B, A, B;
`--in-pcts 40,85 --out-pct 1 --reps 1`, `cache_n 0`):

| Lane / level | 1.1.53 (r1 / r2) | 1.1.55 (r1 / r2) | Mean Δ | VRAM floors (card0 / card1) & RAM |
| --- | ---: | ---: | ---: | --- |
| **DeepSeek** pp 40 % | 1,339.1 / 1,331.4 | 1,331.0 / 1,335.3 | −0.2 % | 1,004–1,014 / 632–645 MiB; `MemAvailable` 14.4–14.7 GiB |
| **DeepSeek** tg 40 % | 27.0 / 27.1 | 26.6 / 27.0 | −0.9 % | — |
| **DeepSeek** pp 85 % | 1,130.8 / 1,125.8 | 1,176.9 / 1,159.9 | **+3.6 %** | plausible source: `DS4_AI_RUN` + `MAYA_DS4_SPEC_HEAD` |
| **DeepSeek** tg 85 % | 27.4 / 27.8 | 27.5 / 27.8 | +0.2 % | — |
| **GLM (derisked)** pp 40 % | 895.6 / 899.8 | 900.2 / 902.5 | +0.4 % | 271–274 / 434–453 MiB (reserve 1200/650); `MemAvailable` 13.1 GiB |
| **GLM (derisked)** tg 40 % | 19.6 / 19.3 | 19.5 / 19.2 | −0.5 % | — |
| **GLM (derisked)** pp 85 % | 925.3 / 932.1 | 937.4 / 932.4 | +0.7 % | — |
| **GLM (derisked)** tg 85 % | 18.8 / 19.8 | 19.7 / 20.2 | +3.4 % | inside ±5 % spread |

# 20261106 GLM 4K picture budget (`vision.max_tokens=12288`, `VISION_FOOTPRINT_MB=4711`) and vanilla Maya-M lane

## GLM picture budget and encoder footprint pin

**Token math**: Qwen3.8-Flash-Next uses patch-16 × merge-2 (32-px cells = 1,024 px/token, so 3840×2160 = 8,160 tokens
fits `8192`). GLM's tower (`glm5next_mmproj.py` + `glm5next_patch.py`) is patch-14 × merge-2 (28-px cells = **784
px/token**): 3840×2160 needs **10,549 tokens** (137×77 grid, 3836×2156 aligned) and 4096×2160 needs 11,244. Without an
override, `serve/server.py` auto-fits 4096/2048/1024 into 15% of the card (picking 4096, downscaling 4K to ~2380×1344).
Set `"max_tokens": 12288` under `"vision"` in the GLM lane config (passed to `maya-vision --max-tokens`).

**Why the measured lend under-covered 12288**: `maya-vision --measure` warms up with a fixed 2048×2048 image (5,329
tokens, 21,316 patches) and never scales up. A real 4K image is 42,196 patches, peaking ~**755 MiB** above the
warm-up's 3.40 GB lend and dipping into the monitor card's reserve:

| State (sysfs `card1` = monitor / CUDA0) | Idle free | 4K picture turn | 9K text prompt |
| --- | ---: | ---: | ---: |
| reserve 1200, measured 3.40 GB lend | 527 MiB | **35 MiB** | 437–453 MiB |
| reserve 1650 / 1950, measured lend | 977 / 1,277 MiB | 222 / 522 MiB | — / 1,193 MiB |
| **reserve 1200 + `VISION_FOOTPRINT_MB=4711` (4.76 GB lend)** | **527 MiB** | **1,164 MiB** | **443 MiB** |

**Fix (`VISION_FOOTPRINT_MB=4711`)**: `deploy/maya/entrypoint.sh` writes `/data/config/vision-memory.json` (keyed by
mmproj path/size/mtime, first GPU, budget, and flash-attn) at every start, so the engine lends **4.76 GB** (`210
experts moved to colder slots, 378 dropped` during the ~15 s encode, restored after) instead of padding idle reserve.
Verified on a 3840×2160 PNG: `prompt_tokens` **10,579** (10,549 image + 30 text, no downscale), 4,569 experts in VRAM
(2,265 + 2,304), floors 1,164 / 349 MiB during the 4K turn and 443 / 279 MiB on text. (Tracked in `notes/TODO.md`:
upstream `strata_vision.cpp` should size its `--measure` warm-up from `max_tokens`.)

## Vanilla Maya-M lane (`glm-5.3-flash-iq2s`)

Provisioned the vanilla `GLM-5.3-Flash-Maya-M-IQ2_S` quant (`peasantsmith/GLM-5.3-Flash-Maya-GGUF`, folder `Maya-M`, 3
shards, 47.68 + 47.82 + 20.50 = 116.0 GB, SHA-256s `3f740d1b…` / `285951d2…` / `ebf1ce71…`) at
`/mnt/unenc-xfs/maya/glm53f-iq2s` using the same per-file `:ro` bind layout and one-shot `--setup` command as the
derisked lane (1,257 tensors, 317 native, 0.88 GiB arena, 1.14 GB vision tower).

`deploy/maya/configs/glm53f-iq2s/maya-glm53f-iq2s-hip.json` (mounted at `/repo-configs-glm-vanilla`) copies the
derisked config 1:1: 512k ctx, `--kv int8`, `--prefill 16384` + `MAYA_GLM_PREFILL_MB=6144`, 36 GB RAM tier, reserves
1200/650, take-two kernels (`h64lvc`/`rl2b4`), 4 × 16 GB disk slots (`KEEP=1`), `vision.max_tokens=12288`, and
`VISION_FOOTPRINT_MB=4711`.

Verified via llama-swap (`glm-5.3-flash-iq2s`, port 10002): ready in 28 s, identical placement (`[0,24)` / `[24,45)`,
4,569 VRAM experts, 35.98 GB RAM tier, NextN MTP on CUDA1), **7/7 gates passed** (chat, tool call, `reasoning_effort:
none`, primes, fenced `sum_list`, 420×420 picture, and 3840×2160 4K picture → 10,579 prompt tokens), min free **445 MiB
monitor / 286 MiB other**, `MemAvailable` min 13.5 GiB, swap +0.08 GiB.
