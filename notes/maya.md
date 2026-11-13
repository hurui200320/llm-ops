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

# 20261107 Maya 1.1.56 upgrade, #861 GLM repack, and 1.1.55 A/B

Upgraded to `v1.1.56` (commit `22fc101f2c109b7e7174c975ff3c2d8e9ce0a867`, 36 commits over `v1.1.55`; llama.cpp
`3cf03257…`, ROCm `7.2.1-complete` base, `gfx1100`, and the `100202` hipBLASLt table unchanged). Workflow run
`42229293542` (30m16s) pushed `ghcr.io/hurui200320/llm-ops-maya:1.1.56-rocm-gfx1100`, digest
`sha256:93723dbc266e3c309281b1f42ed7c4758009f06b0c9612740199f6d305edc7d4` (30,131,364,031 B). Both build gates
passed unchanged: `setup.py`, `maya.py`, `pack_gguf.py`, `vision_pack.py`, `glm5next_patch.py` and `tools/hip/` are
untouched in this release; the only build-relevant change is `tools/iq_pack.py` (#861). The same source/Dockerfile
was also built locally on fedora-tuf first (30,130,492,108 B, `sha256:9c8ca28a…`) and smoke-tested (engines, ldd,
stamps, venv) before the CI image landed; everything below ran on the CI image.

## Upstream changes reaching our image and lanes

- **#860 (GLM, two-GPU splits)**: since v1.1.52 the drafting decode could crash ("illegal memory access") or slow to
  a crawl on some two-GPU PCs; a two-GPU split now drafts as in v1.1.51, and a draft with no valid token can no
  longer read past its table. Our lanes never crashed on 1.1.53/1.1.55 and the 1.1.56 ladders show the same tg and
  acceptance (below). The same release makes `MAYA_GLM_NO_SPEC`/`NO_MTP`/`NO_PREFILL` set to `0`/`false`/`off`/`no`/
  empty mean off (none set in our configs).
- **#861 (packer)**: `iq_pack.py` no longer rounds `exp_probs_b.bias` to BF16; the noaux_tc selection bias (288
  values a layer, 42 layers) stays F32. The engine always read it as F32, so a 1.1.55-written pack loads unchanged
  but keeps the BF16-rounded bias (BF16 left 6-11 distinct values of 288 a layer). Both GLM packs were rebuilt.
- **#845**: a GLM request without `seed` now draws a fresh seed; a request seed (or `--seed`) still repeats exactly.
- **#840**: a GLM conversation whose saved slot fails to load tries the next slot and then the checkpoint.
- **#844**: DeepSeek's tool loop guard now fires on DSML tool calls (before it never did) - automatic, no config.
- **#809**: on AMD, `DS4_AR_P2P` all-reduce buffers stay host-mapped whatever the BAR (the peer buffers hung the
  first request on a pair of RX 7900 XTX - our cards); we never enable it.
- EXL3-only changes (#834/#864 `EXL3_WMMA_SHAPE`, #849/#850/#851 helper knobs, #865/#866 EXL3 in place, #869
  `MAYA_DS4_PROMPT_F16` on gfx103x, #870 `MAYA_DS4_EXPERT_MMQ` IQ1_S) do not apply: our lanes serve GGUF IQ quants
  and have no EXL3 packs or third GPU. `MAYA_GLM_P2P` (#853) now takes any pair reporting peer access (was MI50
  only); measured and rejected (notes/glm/5.3-flash.md 20261107).
- Server: attachment support (#837: PDF/Word/Excel/... through the chat and the OpenAI/Anthropic file parts) lives in
  `serve/attach.py`+`pdf*.py` - stdlib only, no new `setup.py` packages, poppler optional (not installed).

## #861 GLM pack repack (both lanes)

Rebuilt `/data/pack` in both GLM data trees with `tools/iq_pack.py --gguf <shard1> --out /data/pack --compat-bf16`
from the new image (the setup's own invocation; `out.mkdir(exist_ok=True)` keeps `slots/`). Verified on both:
`compat-bf16.json` 849 -> 807 converted tensors (42 `exp_probs_b.bias` gone), the index row's kind 4 (BF16) -> 2
(F32), `dense.bin` 943,911,040 -> 943,935,232 B (+24,192 = 42 x 576), tensors 1,257 unchanged. The pre-repack pack
(index, `dense.bin`, `compat-bf16.json`, `sources.json`, `native_experts.txt`) is kept at
`<data>/pack-backup-1.1.55/`.

## Slots

- **DeepSeek**: 117 conversations / 7.21 GB from the 1.1.55 runs were reloaded by 1.1.56 unchanged. The DS4 session
  identity carries the session ABI (`engine=0.1.27`, CMake `PROJECT_VERSION`), not the release version, and the pack
  did not change.
- **GLM**: the repack invalidates kept slots: the GLM session identity includes `index.txt`/`dense.bin`/
  `native_experts.txt` sizes (plus the same ABI string), and `dense.bin` grew by 24,192 B. 1.1.56 removed the stale
  conversations on first start (derisked 4 / 4.2 GB, vanilla 2 / 227 MB). Without the repack they would have
  survived; a lane bump alone does not clear them.

## Verification (official CI image)

All three lanes, 7/7 functional gates each (chat `SMOKE-OK`; `get_weather` -> `{"city":"Paris"}`;
`reasoning_effort: none` -> `42`; first five primes; fenced `def sum_list`; 420x420 picture; 3840x2160 picture). 4K
pictures: GLM `prompt_tokens` 10,591 (137x77 grid, no downscale; the text part differs from 20261106's 10,579), DS4
1,185 (1140 budget). Ready: DS4 20 s, GLM-derisked 31 s, GLM-vanilla 29 s. Placement: DS4 2,441 + 2,114 = 4,555
experts in VRAM (manual [0,17]/[18,39] split, 36 GB store, 3,512 experts); GLM (both lanes) auto split
[0,24)/[24,45)+MTP, 2,265 + 2,304 = 4,569 experts, RAM tier 17.57 + 18.41 GB, 16K chunks. #845 checked on the GLM
lane: two unseeded sampled requests differ, `seed: 42` repeats exactly.

## Interleaved 512k A/B (1.1.55 vs 1.1.56)

Direct-container ladders on the repacked GLM packs (`--in-pcts 40,85 --out-pct 1 --reps 1`, `cache_n 0`, A/B/A/B):

| Lane / level | 1.1.55 (r1 / r2) | 1.1.56 (r1 / r2) | Mean delta | VRAM floors (card0 / card1) & RAM |
| --- | ---: | ---: | ---: | --- |
| **DeepSeek** pp 40 % | 1,361.4 / 1,352.8 | 1,341.7 / 1,350.6 | -0.8 % | 1,002-1,021 / 626-644 MiB; `MemAvailable` 14 GiB |
| **DeepSeek** tg 40 % | 26.9 / 27.2 | 26.2 / 26.7 | -2.2 % | - |
| **DeepSeek** pp 85 % | 1,183.4 / 1,159.2 | 1,167.8 / 1,137.2 | -1.6 % | - |
| **DeepSeek** tg 85 % | 27.6 / 27.9 | 27.3 / 27.7 | -0.9 % | - |
| **GLM (derisked)** pp 40 % | 900.7 / 898.7 | 902.8 / 903.1 | +0.4 % | 273-286 / 437-449 MiB; `MemAvailable` 12-13 GiB |
| **GLM (derisked)** tg 40 % | 18.9 / 18.9 | 19.1 / 18.8 | +0.3 % | - |
| **GLM (derisked)** pp 85 % | 940.9 / 941.8 | 937.0 / 935.3 | -0.6 % | - |
| **GLM (derisked)** tg 85 % | 19.3 / 17.9 | 18.2 / 20.0 | +2.7 % | - |

All deltas are inside the run-to-run spread (+-3 %); no regression on either lane. GLM draft acceptance stayed in
its usual band (1.1.55 0.740-0.785, 1.1.56 0.700-0.796): the #860 two-GPU drafting restore neither helps nor hurts
this pair.

## Tuning arms

- **`DS4_CHUNK_GFX11=0`** (the new SETTINGS.md recommendation for a discrete RDNA 3 card): the engine still picks
  the borrowed `prompt chunk 16384`, exactly as with our `16384` cap - on this two-GPU split the cap is not the
  binding constraint. Ladder 1,358.7 / 1,180.7 pp and 26.7 / 28.1 tg at 40/85 % = the deployed cap within noise. No
  change; `16384` stays (record: notes/deepseek/v41f1017.md 20261107).
- **`MAYA_GLM_P2P=1`**: no gain (896.8 / 933.5 pp, 19.1 / 17.5 tg) and SETTINGS.md warns consumer-AMD peer access can
  be unreliable (#809 hung on this card pair in the DS4 all-reduce path). Rejected (notes/glm/5.3-flash.md
  20261107).

# 20261110 Maya 1.1.57 upgrade (GLM-5.3 744B bring-up) and 1.1.56 A/B

Upgraded to `v1.1.57` (commit `00655481ce2b0689ab1c2e8d67034dc5701d6511`, 8 commits over `v1.1.56`; llama.cpp
`3cf03257...`, ROCm `7.2.1-complete` base `sha256:3db551c4...`, `gfx1100` and the `100202` hipBLASLt table all
unchanged). Workflow run `42659078538` (31m45s) pushed `ghcr.io/hurui200320/llm-ops-maya:1.1.57-rocm-gfx1100`,
digest `sha256:1372c5d0d33060841cde17951ec3fc4e7708e26e71e425d8128aa941f04a3ee5` (30,141,963,419 B; 1.1.56 was
30,131,364,031 B - the new glm-dsa sources add ~10 MB of binary). Build gates passed unchanged: `setup.py`'s only
change is the setup screen's `model_named()`, `tools/iq_pack.py` only gains a `glm-dsa` architecture branch, and
`tools/ds4/`, `pack_gguf.py`, `vision_pack.py`, `glm5next_patch.py`, `stamps.py` and `tools/hip/gfx1100-*` are
untouched (the new `gfx1151-glm53-hipblaslt-100202.txt` is Strix Halo's).

## Upstream changes reaching our image and lanes

- **GLM-5.3 (744B, glm-dsa) on the same engine**: ~18 cmake fragments compile the new `glm_dsa_*`/`glmd_*` sources
  into `strata_engine` on HIP (build 31m45s vs 30m16s at 1.1.56), the server/dashboard gain its model entry and
  text-only handling, and `MAYA_GLM_*` (= `STRATA_GLM_*` via `maya_names.alias_env`) gains the DSA knobs
  (`MAYA_GLM_DSA_GRAPHS`, `MAYA_GLM_IDX_FP16`, `MAYA_GLM_PROMPT_CACHE_DIR`, `MAYA_GLM_SPLIT_CARDS`) plus per-request
  `strata_tune` engine overrides. It is **not usable on this host**: ~256 GB of VRAM+RAM together for every expert,
  238.6 GB of UD-IQ2_M weights, text only, up to 1M context. Nothing was added to the lanes.
- **Flash-eligible knobs, all default off**: `MAYA_GLM_NGRAM_DRAFT` (+`_MIN`/`_MAX`; history-repetition drafts,
  MTP stays the fallback), `MAYA_GLM_LANE_ROW4` (4-row CPU-lane GEMV, bit-exact, AVX2/FMA/F16C) and the heat-evict
  policy (`MAYA_GLM_VRAM_EVICT=lfu|lru`, default off - "Maya's aged-count policy, unchanged"). The prompt cache is
  hard-gated on `GlmDsa`, i.e. not available to the Flash lanes (checked in `glm_prompt_cache_engine.cpp`).
- **Fixes that reach us**: a SIGTERM / Ctrl+C while the server waits no longer hangs it (78c450a) - lane stops via
  `docker stop -t 120`; DeepSeek video parts now answer a clear 400 instead of an encoder error (serve-side only,
  `src/ds4/` is untouched); Ollama `"think": false` now turns thinking off (we do not use that API); the setup header
  names the model being set up (cosmetic).
- **Shared GLM code**: `glm_model.cu` (auto-split cost from a geometry descriptor; Flash packs fall back to the old
  path), `glm_prefill.cu`/`glm_multi.cu`/`glm_mtp.cu`/`glm_fast_path.cu` and the shared kernels changed for the DSA
  paths. The 1.1.57 ladders and placement below show no behaviour change on our packs. **No repack**: the pack tools
  are untouched and `glm5next` packs are unaffected by the `iq_pack.py` branch; the session ABI stays
  `engine=0.1.27`.

## Verification (official CI image)

All three lanes, **7/7 functional gates** through llama-swap (chat `SMOKE-OK`; `get_weather` -> `{"city":"Paris"}`;
`reasoning_effort: none` -> `42`; first five primes; fenced `def sum_list`; 420x420 picture; 3840x2160 picture):
DS4 first request 27.2 s (cold start + load), GLM derisked 36.1 s, GLM vanilla 32.6 s. 4K pictures:
`prompt_tokens` **10,578** on both GLM lanes (137x77 grid, no downscale) and **1,145** on DS4 (1140 budget).

Slots survived the bump unchanged (no pack change, same ABI): DS4 reloaded **131 conversations / 11.22 GB**,
GLM derisked **4 / 1.66 GB**, GLM vanilla **3 / 4.71 GB**. The derisked lane's first start printed the entrypoint's
config-diff line for the **pre-bump live `env`**; the repo copy replaced it and both GLM live configs now match
`deploy/maya/configs/` (dashboard edits from earlier sessions are reset, nothing to port). Vanilla started clean.
The DeepSeek tune tree was a copy of the production pack without its 131 kept conversations (so the ladder could
not evict them); `ds4.txt` already carried `model_dir /data/model` and the `DS4_VISION_TOKENS=1140` budget.

## Interleaved 512k A/B (1.1.56 vs 1.1.57)

Direct-container ladders on the deployed configs (tune trees, one-shot containers on 127.0.0.1:10110, llama-swap
idle, `--in-pcts 40,85 --out-pct 1 --reps 1`, `cache_n 0`; isl 209,42x / 445,69x), A/B/A/B:

| Lane / level | 1.1.56 (r1 / r2) | 1.1.57 (r1 / r2) | Mean delta | VRAM floors (other / monitor) & RAM |
| --- | ---: | ---: | ---: | --- |
| **DeepSeek** pp 40 % | 1,351.6 / 1,351.7 | 1,342.3 / 1,342.3 | -0.7 % | 1,003-1,017 / 685-696 MiB; `MemAvailable` 14.5-14.8 GiB |
| **DeepSeek** tg 40 % | 27.1 / 27.1 | 26.2 / 26.8 | -2.2 % | - |
| **DeepSeek** pp 85 % | 1,186.7 / 1,146.9 | 1,165.8 / 1,181.7 | +0.6 % | - |
| **DeepSeek** tg 85 % | 27.5 / 27.1 | 27.3 / 27.4 | +0.2 % | - |
| **GLM (derisked)** pp 40 % | 978.0 / 977.5 | 978.1 / 978.4 | +0.05 % | 275 / 488 MiB; `MemAvailable` 13.0-13.1 GiB |
| **GLM (derisked)** tg 40 % | 19.0 / 19.5 | 19.3 / 18.4 | -2.1 % | - |
| **GLM (derisked)** pp 85 % | 1,004.5 / 996.4 | 1,000.5 / 999.1 | -0.06 % | - |
| **GLM (derisked)** tg 85 % | 17.8 / 21.0 | 19.8 / 19.6 | +1.5 % | - |

All deltas are inside the run-to-run spread (the GLM 85 % tg spread alone is 17.8-21.0 on one image); no
regression. Placement is identical arm for arm: GLM auto split `[0,24)`/`[24,45)`, 16K chunks, prestage 160
experts, 36 GB RAM tier (17.57 + 18.42 GB); DS4 `DS4_SPLIT=18,22`, borrowed `prompt chunk 16384`, 3,512 RAM
slots (10,806-10,820 experts in place). Swap stayed flat on the GLM ladders; the DS4 85 % prompts dip it
(51-67 GiB free at the minimum on both versions - page-cache pressure from the 240 GB of shards, same on 1.1.56).

## Tuning arms (GLM, 1.1.57)

Screened at 40 % (1 rep, one-shot containers, base control interleaved; the base band across the day's runs is
976.3-978.4 pp / 18.4-19.5 tg, with one 962.6/17.4 outlier):

| Arm | pp 40 % | tg 40 % | verdict |
| --- | ---: | ---: | --- |
| Base (control) | 976.3 | 19.5 | - |
| `MAYA_GLM_NGRAM_DRAFT=1` | 976.9 | 19.4 | no gain on book text (no repetition) |
| `MAYA_GLM_LANE_ROW4=1` | 976.8 | 19.4 | no gain |

- **`NGRAM_DRAFT`**: a verbatim-copy test (1,437-token text block repeated from the prompt, 1,399 tokens emitted)
  shows why it has no headroom here: the MTP block drafted 1,397 tokens and 1,395 were accepted (99.9 %) in **both**
  arms (23.8-23.9 tg), and the n-gram session stood down - `pick()` prefers the MTP draft when the n-gram's first
  token is not better, so no `glm ngram:` drafts were taken. Left off.
- **`LANE_ROW4`**: bit-exact, but the CPU lane is not on the critical path: the copy test's decode stat is 41.9
  ms/token with the CPU lane at 69.9 experts/token in 25.9 ms, overlapping the 24.3 ms/token of disk waits. Left off.
- **`VRAM_EVICT`**: not tested on purpose - tg is bounded by the RAM expert tier and the SSD reads, and the knob only
  changes which experts VRAM evicts (the user's call, 2026-11-10).

# 20261111 Maya 1.1.58 upgrade and 1.1.57 A/B

Upgraded to `v1.1.58` (commit `f4067232fa599494122508b55556c4dbaad945ba`, 2 commits over `v1.1.57`; llama.cpp
`3cf03257...`, ROCm `7.2.1-complete` base `sha256:3db551c4...`, `gfx1100` and the `100202` hipBLASLt table all
unchanged). Workflow run `42977699783` (30m45s) pushed `ghcr.io/hurui200320/llm-ops-maya:1.1.58-rocm-gfx1100`,
digest `sha256:f8149c180f0e7e74d3395abe1d3120c6760ea877774f7d493c59bb2724ae0a14` (30,142,539,960 B; 1.1.57 was
30,141,963,419 B - the new `glm_ablate.cu` and test code). Build inputs passed the workflow gates unchanged:
`setup.py`, `maya.py`'s build path (`check_hip_pc`, `compile_engine_hip`/`compile_vision`, stamps), `iq_pack.py`,
`pack_gguf.py`, `vision_pack.py`, `glm5next_patch.py`, `tools/hip/gfx1100-*` and `tools/ds4/` are untouched;
`CMakeLists.txt` only adds `src/core/glm_ablate.cu` to `strata_engine` (+ its test).

## Upstream changes reaching our image and lanes

- **No pack change, no ABI change**: the GLM-5.3 Q8_0-attention pack change is glm-dsa only
  (`tools/strata/glm_dsa_pack.py`; `headwise_stale` returns None for `glm5next` packs), `iq_pack.py` (the Flash
  packs' tool) is untouched, and the session ABI stays `engine=0.1.27` - both GLM packs and the DS4 pack keep their
  slots.
- **`LANE_ROW4` default on** (`STRATA_GLM_LANE_ROW4`, the 4-row CPU-lane GEMV, bit-exact): the only default that
  flips on our GLM lanes (screened neutral on 1.1.57). Kept; configs unchanged.
- **Loop guard (#921/#925)**: an answer to a user message (not a tool result) is never kept going, and a kept
  agent answer continues on a new paragraph. Server-side only, automatic - it reaches our chat/agent use.
- **FP16 expert prompt path** (IQ1_M / Q4_0 / Q4_1 / Q5_0 / Q5_1 in `glm_prefill.cu`, glm-dsa-shaped) and the
  **Q5_K multi-token GEMV** (`glm_fast.cu`, bit-exact): our Flash packs' types (IQ2_S / IQ3_XXS / Q6_K / Q3_K /
  Q4_K) are already MMQ/GEMM-covered, so no effect measured or expected.
- **DS4 #920/#697**: the one-GPU NVIDIA decode-streaming path only; our 2-GPU `DS4_SPLIT=18,22` lane never takes
  the path. The ablate groundwork (`--ablate*`, `GEN ablate=`, server/chat plumbing) stays off without a direction
  file (none is published).
- New opt-in `MAYA_GLM_PREFILL_STABLE=1` (reproducible prompt products whatever the expert cache holds, 2-5 % pp
  cost): a bench-methodology knob, not deployed (notes/TODO.md).

## Verification (official CI image, on the runner)

Pulled on fedora-tuf with the digest above. All three lanes, **7/7 functional gates** through llama-swap (chat
`SMOKE-OK`; `get_weather` -> `{"city":"Paris"}`; `reasoning_effort: none` -> `42`; first five primes; fenced
`def sum_list`; 420x420 picture; 3840x2160 picture): DS4 ready 24.8 s, GLM derisked 43.3 s, GLM vanilla 41.6 s
(first request, cold). 4K pictures: `prompt_tokens` **10,577** on both GLM lanes (137x77 grid, no downscale) and
**1,171** on DS4 (1140 budget). Slots survived the bump unchanged: DS4 reloaded **134 conversations / 11.51 GB**,
GLM derisked **4 / 1.43 GB**, vanilla **4 / 3.18 GB**. No entrypoint config-diff line on either GLM lane (the live
configs already matched `deploy/maya/configs/`). Engine build: `maya-ds4 1.1.58 (build f4067232fa59...)`.

## Interleaved 512k A/B (1.1.57 vs 1.1.58)

One-shot containers on the tune trees, deployed configs, 127.0.0.1:10110, llama-swap idle, A/B/A/B,
`--in-pcts 40,85 --out-pct 1 --reps 1`, `cache_n 0`; isl 209,42x / 445,69x. Artifacts (local, gitignored):
`bench/results/longctx/maya-ab-20261111/`.

| Lane / level | 1.1.57 (r1 / r2) | 1.1.58 (r1 / r2) | Mean delta | VRAM floors (other / monitor) & RAM |
| --- | ---: | ---: | ---: | --- |
| **DeepSeek** pp 40 % | 1,364.8 / 1,241.2 | 1,261.5 / 1,365.3 | +0.8 % | 809-821 / 561-566 MiB; `MemAvailable` 12.4-13.2 GiB |
| **DeepSeek** tg 40 % | 27.4 / 27.2 | 26.5 / 26.5 | -2.9 % | - |
| **DeepSeek** pp 85 % | 1,158.9 / 1,177.4 | 1,181.2 / 1,183.9 | +1.2 % | - |
| **DeepSeek** tg 85 % | 27.4 / 27.4 | 27.0 / 27.3 | -0.9 % | - |
| **GLM (derisked)** pp 40 % | 974.1 / 976.8 | 977.3 / 977.7 | +0.2 % | 73 / 359 MiB; `MemAvailable` 10.5-11.3 GiB |
| **GLM (derisked)** tg 40 % | 19.0 / 19.9 | 19.5 / 18.7 | -1.8 % | - |
| **GLM (derisked)** pp 85 % | 1,001.9 / 1,004.2 | 995.8 / 999.0 | -0.6 % | - |
| **GLM (derisked)** tg 85 % | 18.2 / 20.4 | 19.0 / 19.5 | -0.3 % | - |

All deltas are inside the run-to-run spread (+-3 %; the GLM 85 % tg spread alone is 18.2-20.4 on one image); no
regression on either lane. Placement is identical arm for arm: GLM auto split `[0,24)`/`[24,45)`, 16K chunks,
prestage 160, 2,265 + 2,304 experts, 36 GB RAM tier (17.57 + 18.41 GB); DS4 `DS4_SPLIT=18,22`, borrowed prompt
chunk 16384, 3,512 RAM slots, 4,541-4,554 experts in VRAM. GLM draft acceptance stayed in its band (1.1.57
0.771-0.786, 1.1.58 0.763-0.803): the new default `LANE_ROW4` is bit-exact and moves nothing measurable here. Swap
was near-full during the DS4 85 % prompts on both versions (page-cache pressure from the shards, as before).



# 20261112 Maya 1.1.59 upgrade, refusal ablation and the ablation A/B

Upgraded to `v1.1.59` (commit `027992cb1f09829488e650e466233ca351d3d6ef`, 3 commits over `v1.1.58`; llama.cpp
`3cf03257...`, ROCm `7.2.1-complete` base, `gfx1100` and the `100202` hipBLASLt table all unchanged). GitHub CI
failed three times on a runner-side DNS outage (`Could not resolve 'security.ubuntu.com'` inside the Dockerfile's
own `apt-get update`, a step every build runs), so the image was built on fedora-tuf with the workflow's exact
build args (24-core box, ~12 min: warm base layers, engines + vision recompiled from the new source). Smoke tests
replicated the workflow's seven in-image gates (engine `--help`/`--version`, vision usage, `ldd`, stamps
`src=95fc0c552f73ea91`, venv imports, `maya.py --help`) - all pass; `maya-ds4 1.1.59 (build 027992cb...)`. Image
`ghcr.io/hurui200320/llm-ops-maya:1.1.59-rocm-gfx1100`, manifest list `sha256:9c027bf37f0b322a7612d19ece91d6fb25013861c2ad0a54669ead44ecb27806`
(built locally; no GHCR push from the runner - no docker login there; CI can publish the equivalent artifact when
its network recovers).

## Upstream changes reaching our lanes

- **Internal `STRATA_*` -> `MAYA_*` rename pass** (the bulk of the 365-file diff): env vars, tuning-table headers
  (`STRATA_HIPBLASLT_TUNING_V1` -> `MAYA_HIPBLASLT_TUNING_V1`, table + engine ship in one image), cmake targets.
  All 21 env vars our lanes deploy keep their names (checked one by one); `VISION_FOOTPRINT_MB` is ours.
- **No pack change, no ABI change** (`project(strata VERSION 0.1.27)` unchanged, `iq_pack.py`/`pack_gguf.py`/
  `vision_pack.py` untouched): no pack rebuilds, all slots survive. Verified: DS reloaded 134 conversations /
  11.51 GB, GLM vanilla 4 kept.
- **Refusal ablation ships** (groundwork was 1.1.58): per-request direction projection for maya-ds4 and the GLM
  engine, Maya's direction files published in each model's Maya repo, `tools/ablate_fit.py` gains `eval`. See the
  ablation section below and deploy/maya/README.md.
- Setup quant/context recommendation rework, GLM-5.3 (744B) Turbo (`MAYA_GLM_ROUTE_AFFINITY`), the #883 fair
  slices fix (`--parallel`, not deployed), #884 (`DS4_LENT_RAM_GB`, not used), `MAYA_DS4_HIP_PEER_AR`
  (NVIDIA-only): none touch our lanes. `MAYA_GLM_ROUTE_AFFINITY` (lossy, changes answers) is a candidate future
  tuning arm for the GLM lanes - notes/TODO.md.

## Refusal ablation wiring

- **Direction files** (control-vector GGUFs, one F32 `direction.1` of the hidden size): DeepSeek
  `peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF` -> `DeepSeek-V4.1-Flash-refusal-direction.gguf`
  (21,024 B, sha256 `09ab0ac8...`, fitted on Maya-S, layer 25, dim 5120, scale 1.0, think off); GLM-5.3-Flash
  `peasantsmith/GLM-5.3-Flash-Maya-GGUF` -> `GLM-5.3-Flash-refusal-direction.gguf` (16,928 B, sha256
  `ecf89232...`, fitted on Maya-S24, layer 31, dim 4096). Downloaded once per lane tree into
  `<tree>/refusal-directions/` (= `/data/refusal-directions/` in-container); the derisked GLM lane carries none
  on purpose (its quant is already an abliterated transplant - projecting would ablate it twice).
- **Config wiring**: the server does NOT inject the flags at spawn - `maya.py`'s setup-time `ablate_step` bakes
  `--ablate FILE --ablate-default off` into the config's `args`, and the serve path never re-derives them (the
  `models.apply` injection runs only on dashboard saves). So the lane configs carry both the `"ablate"` key
  (file + fitted_on, what the Models page reads) AND the two flags in `args`. First attempt with the key alone
  started the engine without a direction (`/props` said `available: false`); the flags fixed it.
- **llama-swap alias**: per-lane `filters.setParamsByID` `"${MODEL_ID}:ablation"` -> `ablate: true` (the `:npt`
  pattern; the key auto-registers as an alias, the middleware merges the top-level field into any JSON body, and
  maya never validates the body's `model` field). Default stays off everywhere: `--ablate-default off` + no
  `ablate.default` key, so plain requests are bit-identical to no direction file.
- **Verification** (all on 1.1.59): 7/7 functional gates per lane (chat, tool call, reasoning_effort none,
  primes, fenced code, 420x420 and 4K pictures; DS 4K `prompt_tokens` 1,166 = its 1140 budget, GLM 10,572 = no
  downscale), and the vanilla GLM + DS lanes repeat **7/7 through the `:ablation` alias** (tool calls,
  reasoning_effort and 4K pictures all flow through the injected field). `/props` `ablate.available: true`
  (default false) on DS + vanilla; `false` on derisked. DS slots 134 / 11.51 GB, GLM vanilla 4 - both lanes
  kept their conversations across the bump.

## DeepSeek: the Maya-S direction transfers to the ISTA quant (tools/ablate_fit.py eval)

`ablate_fit.py prompts` (pinned AdvBench + Alpaca, 416/104 train/test harmful, 416/104 harmless, 16 coherence)
then `eval --model dsv41f1017-iq2xxs --harmless` against the production lane (temperature 0, think off - the
file's own fitting condition), ~37 min for 448 requests:

| | ablation off | ablation on |
| --- | ---: | ---: |
| held-out harmful refusals | 99.0 % | **3.8 %** |
| harmless false refusals | 0.96 % | 0.0 % |
| coherence loops / empty / refusals | 0 / - / - | 0 / 0 / 0 |
| coherence length ratio | - | 1.02 |

`verdict: "ok"`. Upstream's own DeepSeek numbers (their quant): 98 % -> 1 %; the 2.8-point residual here is
plausibly the IQ2_XXS quant or run variance - at 3.8 % the direction transfers. Sets + JSON:
`/mnt/unenc-xfs/maya/ablate-eval/` on fedora-tuf (not in the repo).

## Ablation A/B (512k, deployed configs, off = `--ablate-default off`, on = `on`)

One-shot containers on the tune trees (DS: `dsv41f1017-iq2xxs-tune`; GLM: a fresh `glm53f-iq2s-tune` copy of the
vanilla tree minus slots - the derisked tune tree cannot serve an ablation arm), direct on 127.0.0.1:10110,
llama-swap idle, A/B/A/B interleaved, `--in-pcts 40,85 --out-pct 1 --reps 1`, `cache_n 0`, vram_sampler at 5 s.
Both arms load the direction; only `--ablate-default` differs, so the delta is exactly the projection's cost.
Driver: `/mnt/unenc-xfs/maya/tune/ab59_point.sh`; artifacts `/mnt/unenc-xfs/maya/tune/ab59/results/`.

**DeepSeek (2 clean reps per arm, floors card0/card1 669-687 / 505-516 MiB, `MemAvailable` 12.9-13.2 GiB, swap
near-flat):**

| Lane / level | off (r1 / r2) | on (r1 / r2) | mean delta |
| --- | ---: | ---: | ---: |
| **DS** pp 40 % | 1,315.0 / 1,349.6 | 1,349.4 / 1,287.8 | -1.0 % |
| **DS** tg 40 % | 26.60 / 26.90 | 26.10 / 26.30 | -2.1 % |
| **DS** pp 85 % | 1,183.1 / 1,168.2 | 1,194.1 / 1,181.0 | +1.0 % |
| **DS** tg 85 % | 27.40 / 27.20 | 27.40 / 27.20 | 0.0 % |

All deltas inside the +-3 % spread - the DS projection (a per-device direction copy, projected at each
`hc_post`) costs nothing measurable. **The alias is free on this lane.**

**GLM vanilla:** the first round's two r1 points are outliers from cold-start memory pressure (`MemAvailable`
193/257 MiB, ~48 GB swap churn - fresh tune-tree expert profile plus the DS ladder's page cache): off-r1 85 %
fell to 726.1 pp / 12.8 tg and on-r1 40 % to 10.8 tg. The warm r2 pair: off 973.1 / 18.7 and 990.5 / 18.0 vs on
967.7 / 16.8 and 992.0 / 17.0 - pp unchanged (±0.6 %), tg -10.2 % (40 %) and -5.6 % (85 %) on single warm reps.
A second interleaved round (r3/r4) ran to firm this up - see below.

**GLM vanilla, second round (r3/r4, fully warm - floors 201-239 / 352-372 MiB, `MemAvailable` ~11.7 GiB, swap
flat on all four points):** off 970.3 / 17.8 and 990.5 / 17.2, then 971.7 / 18.4 and 976.4 / 20.5 vs on 969.0 /
17.7 and 988.7 / 17.6, then 969.9 / 18.2 and 991.9 / 18.4. Pooled with the warm r2 pair (3 reps per arm):

| GLM vanilla / level | off (warm) | on | mean delta |
| --- | ---: | ---: | ---: |
| pp 40 % | 971.7 (973.1 / 970.3 / 971.7) | 968.9 (967.7 / 969.0 / 969.9) | -0.3 % |
| tg 40 % | 18.30 (18.7 / 17.8 / 18.4) | 17.57 (16.8 / 17.7 / 18.2) | **-4.0 %** |
| pp 85 % | 985.8 (990.5 / 990.5 / 976.4) | 990.9 (992.0 / 988.7 / 991.9) | +0.5 % |
| tg 85 % | 18.57 (18.0 / 17.2 / 20.5) | 17.67 (17.0 / 17.6 / 18.4) | **-4.8 %** |

**GLM verdict: pp free, tg -4..-5 % (about -0.8 t/s) on ablated requests.** Consistent across both rounds' warm
points; the r2 pair's -10.2 % (40 %) was single-rep noise. Draft acceptance is unchanged between arms (0.71-0.80
both), so the cost is the projection itself - the GLM engine projects every residual write of every main layer on
every path (decode token, verify windows, batched rows, both cards), which lands entirely on the decode step.
Floors and placement are identical arm for arm (auto split, 16K chunks, prestage 160, 36 GB RAM tier). The cost
applies only to requests that ask for ablation: plain requests carry nothing on their engine line and are
bit-identical to no direction file, which the off arms confirm (they sit on the production baseline).

Artifacts (local, gitignored): `bench/results/longctx/maya-ab-20261112/` (12 points: 4 DS + 8 GLM, each with
`.json` results, `.log`, `.csv` VRAM/RAM/swap samples and the engine log).

# 20261112 Maya 1.1.60 upgrade and 1.1.59 parity gates (A/B deferred)

Upgraded to `v1.1.60` (commit `8b0a449c599fbb370a1dcc0979ae7c84d4310529`, 3 commits over `v1.1.59`: the release
commit plus two docs commits publishing the direction files; llama.cpp `3cf03257...`, ROCm `7.2.1-complete` base,
`gfx1100` and the `100202` hipBLASLt table all unchanged). Workflow run `43178050443` (25m53s) pushed
`ghcr.io/hurui200320/llm-ops-maya:1.1.60-rocm-gfx1100`, digest
`sha256:7f0dfef0f632f9b756f71c7fe732b444a019e9be5e7ada1ddb38e317965c7f29` (30,142,642,289 B; 1.1.58 was
30,142,539,960 B - the GLM engine sources changed, vision unchanged). Build gates passed unchanged: `setup.py`,
`maya.py`'s build path (`check_hip_pc`, `compile_engine_hip`/`compile_vision`, stamps), `CMakeLists.txt`,
`iq_pack.py`, `pack_gguf.py`, `tools/ds4/pack_gguf.py`, `vision_pack.py`, `glm5next_patch.py` and
`tools/hip/gfx1100-*` are untouched; only the GLM engine's source hash moved (stamp `src=ca1104c6e2bb366d`) with
the ring/DFlash2 changes.

## Upstream changes reaching our image and lanes

- **HIP routing-ring sizing** (the one change that reaches us): since v1.1.57 the host-mapped routing ring was
  sized for GLM-5.3 (744B) - 256 entries x 8192 floats, 8.4 MB; HIP builds now size it per model
  (`ring_entries_for`/`ring_stride_for`): GLM-5.3-Flash 64 x 4096 floats (1 MB), glm-dsa 128 x 6144. Upstream
  measured up to ~0.8% faster drafting decode on gfx1151 APUs, "back to v1.1.56's speed", with identical answers;
  a kernel guard refuses an `n_embd` wider than the ring's entries. Automatic - no config change. Our
  1.1.56 -> 1.1.59 ladders never showed the regression on gfx1100 (an APU phenomenon), so the expected effect here
  is neutral to slightly-positive tg on the GLM lanes.
- **DFlash2 drafting for GLM-5.3 (744B)** (`--dflash` no longer refused on glm-dsa, plain-path capture hooks in
  `glm_residual.cu`, `dflash_pipelined` plain mode): glm-dsa only, not usable on this host (744B); the hooks sit on
  the plain (mHC-less) path our Flash lanes never take. No effect.
- **Setup-side only**: turboderp's EXL3 2.25 bpw DeepSeek quant in the setup (`--ds4-quant exl3-2.25`, ~160 GB of
  VRAM+RAM - fits our 110 GB host worse than the IQ2_XXS GGUF; not deployed), the `APU_CONTEXT_SAFETY_GB=4`
  context-recommendation margin on unified-memory APUs (we pin `--max-context`), direction-file downloads now from
  the model repo's `refusal-directions/` folder with the top-level fallback (our manually downloaded files are
  byte-identical and stay valid), `serve/model_pull.py`'s "Other quants" list. `tools/ds4/pack.py` changes serve
  the EXL3 safetensors packs, not our GGUF lane (`pack_gguf.py` is untouched).
- **No pack change, no ABI change** (`project(strata VERSION 0.1.27)` unchanged): no repacks, all slots survive.

## Verification (official CI image, on fedora-tuf)

In-image smoke gates replicated after the pull (engines `--help`/`--version`, `ldd`, stamps `src=ca1104c6e2bb366d`
/ vision `src=3485b5bec679d4fa`, venv imports, `maya.py --help`) - all pass; `maya-ds4 1.1.60 (build 8b0a449...)`.
Deployed by `git pull` on fedora-tuf (`b5153d7`); llama-swap's `-watch-config` picked up the new tag. All three
lanes, **7/7 functional gates** through llama-swap (chat `SMOKE-OK`; `get_weather` -> `{"city":"Paris"}`;
`reasoning_effort: none` -> `42`; first five primes; fenced `def sum_list`; 420x420 picture; 3840x2160 picture):
4K `prompt_tokens` **1,175** on both DS lanes (their 1140 budget) and **10,581** on the GLM lane (137x77 grid, no
downscale). The GLM vanilla lane repeats **7/7 through the `:ablation` alias** and `/props` reports
`ablate.available: true, default: false`. Slots survived the bump unchanged (no pack change, same ABI): DS
**135 conversations / 11.52 GB**, GLM **4 / 2.00 GB**, abiray **1 / 0.16 GB**. No entrypoint config-diff line on
any lane.

## Interleaved 512k A/B: deferred

The 1.1.59-vs-1.1.60 ladder was **not run**: dev containers occupied the host during the upgrade window and a
benchmark would have measured them, not the release. The only engine change reaching the lanes is the routing-ring
sizing (upstream: <= ~0.8% tg on APUs, identical answers; the 1.1.57 regression it fixes never reproduced on
gfx1100), so the regression risk is low - the lanes go to production on the gate results alone. Tracked in
notes/TODO.md; run the interleaved ladder when the host is idle.

# 20261114 Maya 1.1.61 upgrade: DeepSeek prefill +8.6% at 85% on gfx1100

Upgraded to `v1.1.61` (commit `f66a913c628c6672795088a56ad85de0efc827e4`, 4 commits over `v1.1.60`: three HIP EXL3
decode commits plus the release commit; llama.cpp `3cf03257...`, ROCm `7.2.1-complete` base, `gfx1100` and the
`100202` hipBLASLt table all unchanged). Workflow run pushed
`ghcr.io/hurui200320/llm-ops-maya:1.1.61-rocm-gfx1100`, digest
`sha256:83994e79aa49139fe333638f0d9a9665588ac924215450ac4931271129dc896d` (30,146,959,474 B; 1.1.60 was
30,142,642,289 B - the ds4 kernel changes). Both build gates passed unchanged (`setup.py`'s llama.cpp pin, the
`gfx1100-glm-hipblaslt-100202.txt` table); pack tools (`iq_pack.py`, `pack_gguf.py`, `vision_pack.py`,
`glm5next_patch.py`, `stamps.py`, `tools/hip/`) and the GLM/vision engine sources are untouched in the diff - no
repack, no Dockerfile/entrypoint changes, `project(strata VERSION 0.1.27)` unchanged (slots survive).

## Upstream changes reaching our image and lanes

- **RDNA3 DeepSeek prefill, the release's headline and the reason to take it**: `set_tc_kernel` gains an RM rows
  multiplier (a block takes RM x 64 rows, an expert's weight tile decoded once for all of them; "the same MMAs over
  the same K steps: the same bits") defaulting to **4 on gfx11** (`tc_rows_mult()`; `DS4_IQ_TC_RM=1|2|4` override),
  and `sparse_attn_wmma_kernel` gains a G groups form (G x 16 heads per block, one staged tile for all, bank-padded
  rows; "the same values: the same bits") taking **mode 4 on gfx11 for prompt chunks >= 64 queries**
  (`attn_wmma_mode()`; `DS4_ATTN_SHARE=0|1|4` override). Upstream measured pp +14-17% (Maya-XS, 2K-128K) and +8-9%
  (Maya-S, 8K-32K) on a Ryzen AI Max APU, "other RDNA3 GPUs (Radeon RX 7000) use the same code" - our 2x RX 7900
  XTX qualify. Both changes are automatic, no config.
- **Long-prompt skip refactor** (`ds4_model.cu`): the single cut became a per-layer cut (`ck`/`cr` by layer)
  serving the new cone mode; with the cone off the plan returns the same single cut - behavior-preserving for us.
- **`fast_prefill` / `--fast-prefill`** (Models page switch, `serve/server.py` maps it to `DS4_YOCO_CONE=1`): the
  skip as a per-layer cone - **requires `yoco_one_gpu()`**, so it is unavailable on our 2-GPU `DS4_SPLIT=18,22`
  lanes; also "answers can shift slightly". Not deployed.
- **`MAYA_DS4_LOOKUP_TRUST`** (default on) + per-request `lookup_trust`: a trusted long-match prompt-lookup round
  is verified without drafting. Inert here: our lanes run `--suffix-draft 0` (lookup off). `MAYA_MM3_LOOKUP_TRUST`
  and mm3 `--suffix-draft 3` default: no MiniMax lane.
- **Setup-side only**: the DSpark speed calibration now weighs chat, code and tool calls alike
  (`tools/calibrate_ds4.py`, `maya.py`) - we hand-write configs, so no effect; side thought (not deployed): the
  code-aware calibration + trust machinery could justify re-testing `--suffix-draft` on the DS lane for code-heavy
  traffic. EXL3 decode gains (full-rate shifts/codebook multiply/`EXL3_DECODE_INLINE`) and the maya-mm3 CMake link
  change: no EXL3 packs, `maya-mm3` is not built by our image.

## Verification (official CI image, on fedora-tuf)

In-image smoke gates replicated after the pull (engines `--help`/`--version`, vision usage, `ldd`, stamps
`src=651b9f33b57605a1` / vision `src=19084eccaa9c2513`, venv imports, `maya.py --help`) - all pass;
`maya-ds4 1.1.61 (build f66a913c...)`. Deployed by `git pull` (`9daba5e`); llama-swap's `-watch-config` reloaded
the three lane tags. All three lanes, **7/7 functional gates** through llama-swap (chat `SMOKE-OK`; `get_weather`
-> `{"city":"Paris"}`; `reasoning_effort: none` -> `42`; first five primes; fenced `def sum_list`; 420x420 picture;
3840x2160 picture): 4K `prompt_tokens` **1,146 on both DS lanes** (within the 1140 image budget) and **10,579** on
the GLM lane (137x77 grid, no downscale), plain and through the `:ablation` alias (6/6 + 4K through the alias
too). `/props`: GLM vanilla `ablate.available: true, default: false`; the DS lanes carry no direction by design
(uncensored abiray weights since 2026-11-12). Slots across the bump: abiray **215 conversations / 9.26 GB**, GLM
**4 / 1.50 GB**; the main DS lane keeps **9 / 1.15 GB** and removed 110 as foreign/incomplete - the 11-13 abiray
bench traffic evicted its older conversations under the 16 GB slot cap (not a version effect: the bump kept
abiray's 215). Live `/data/config` diffs clean against `deploy/maya/configs/` on all three lanes after the
restarts.

## Interleaved 512k A/B (1.1.60 vs 1.1.61, DeepSeek lane)

Tune tree recreated first (`/mnt/unenc-xfs/maya/dsv41f1017-iq2xxs-tune`: rsync of the production tree minus
`slots/`, 935 MB, carrying the pack, `expert-profile.bin`, `refusal-directions/`); the ab59-era trees had been
cleaned up. One-shot containers on 127.0.0.1:10110 with the production config folder
(`deploy/maya/configs/dsv41f1017-iq2xxs`), repo entrypoint bind, A/B/A/B interleaved,
`run_longctx.py --ctx 524288 --in-pcts 40,85 --out-pct 1 --reps 1`, `cache_n 0`; vram_sampler at 5 s. Driver:
`/mnt/unenc-xfs/maya/tune/ab61_point.sh` + `ab61_run.sh`; artifacts `/mnt/unenc-xfs/maya/tune/ab61/`.

| Lane / level | 1.1.60 (r1 / r2 / r3 / r4) | 1.1.61 (r1 / r2 / r3 / r4) | Mean delta |
| --- | --- | --- | ---: |
| **DS** pp 40 % | 1286.3 / 1126.2 / 1262.4 / 1263.9 | 1273.3 / 1198.2 / 1375.6 / 1308.1 | **+4.4 %** (1234.7 -> 1288.8) |
| **DS** tg 40 % | 25.7 / 25.5 / 25.8 / 26.5 | 25.2 / 25.5 / 26.2 / 26.3 | -0.3 % |
| **DS** pp 85 % | 1147.4 / 1192.5 / 1144.7 / 1139.4 | 1260.9 / 1263.0 / 1235.7 / 1260.6 | **+8.6 %** (1156.0 -> 1255.1) |
| **DS** tg 85 % | 26.3 / 27.1 / 27.2 / 26.9 | 26.7 / 27.1 / 27.2 / 27.2 | +0.7 % |

The 85% gain (+8.6%, every 1.1.61 point above every 1.1.60 point) matches upstream's +8-9% claim for this size
class; 40% is +4.4% with overlapping spreads (1.1.60's 40% spread alone is 1126-1286). tg is unchanged. Placement
is identical arm for arm (36 GB RAM store, 3,512 slots, 10,803-10,804 CPU-lane experts in place); VRAM floors
838-847 / 609-612 MiB, `MemAvailable` 12.3-13.0 GiB. Both AGENTS.md targets hold with margin (pp >= 1000, tg >= 25
at 85%). No repack, no config change.

## Bit-identity probe: inconclusive, lane output is not boot-reproducible

Each point runs a fixed greedy `seed 42` probe before the ladder. Result: outputs differ **within the same
version** boot-to-boot (1.1.60 r3 vs r4 diverge at byte 150 of 374; 1.1.61 r3 vs r4 at byte 0 - one 1.1.61 boot
answered with a stray meta preamble, "? Desired oververbosity 5."), while one cross-version pair is
byte-identical (1.1.60 r3 == 1.1.61 r4). So the probe cannot attribute drift to the release: this lane's output
varies boot-to-boot regardless of version (the analogue of GLM's expert-cache-dependent sums - for DS4 the
boot-varying expert admission/placement is the likely source; the placement counts stay identical, the sums'
association order need not). The upstream "identical answers" claim is untestable this way here; practically the
gates' canonical answers are unchanged and the 4K/tool/thinking paths all pass. If a reproducibility knob ships
for DS4 (like GLM's `MAYA_GLM_PREFILL_STABLE`), revisit.

## Method notes

- First pass ran with rep-only tags (`r1`, `r2`), so the B arms overwrote the A arms' per-point files
  (`r1.json`, bits, csv, engine logs); the summaries in `run.log` survived and the table above pools both passes.
  Fixed to `<arm>-<rep>` tags for the r3/r4 pass.
- The first launch collided with the still-up GLM lane (llama-swap keeps the last-used lane loaded, `globalTTL:
  0`): the arm's engine died at load behind its `--rm` container and the point script burned its health timeout.
  The point script now checks both cards' free VRAM before `docker run` (a lane woken between arms aborts the
  point instead of measuring a contended arm) and asserts the arm's engine version at start.
