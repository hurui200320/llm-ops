# 20261104 Project Maya on gfx1100: image, DeepSeek setup, strata -> maya switch, and 1.1.53 bump

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

## Pins (`v1.1.52`) and image architecture

| What | Value |
| --- | --- |
| Maya | tag `v1.1.52` (`mw00/project-maya`), commit `3c935e411cd6625a9af0f292dd0af52598e866d8` |
| llama.cpp | `3cf03257f219afbe7334045ff7c6a06ac68c627d` (gated against `setup.py` by the workflow) |
| ROCm base | `docker.io/rocm/dev-ubuntu-24.04:7.2.1-complete` (`sha256:3db551c4e1229aac1857ac44fcb6141bb749f41348eb572452f11279153c13c3`) |
| Arch / hipBLASLt | `gfx1100` only; `tools/hip/gfx1100-glm-hipblaslt-100202.txt` (shipped alongside `100401`, `100500`) |

- **Why the full ROCm dev image**: `check_hip_pc()` runs at **every start** and requires `hipcc` ≥ 7.0 at
  `<root>/bin/hipcc`, ROCm's clang, `libhipblas.so*`, `c++` on `PATH`, AVX2, and ≥1 KFD GPU (`<root>` resolves
  `ROCM_PATH` → `/opt/rocm` → `/opt/rocm-7.2.1`). A libs-only runtime image cannot start Maya, and TheRock wheels
  (`7.10.0a20251120`) do not match Maya's ROCm 7.2.1 / hipBLASLt 1.2.2 (`100202`) table.
- **Engine & vision builds (`deploy/maya/Dockerfile`)**:
  - Engine (`compile_engine_hip`): `-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF
    -DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON
    -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=<rocm clang++> -DCMAKE_PREFIX_PATH=/opt/rocm
    -DSTRATA_GGML_DIR=<llama.cpp>`.
  - Vision (`compile_vision`): runs `tools/vision/glm5next_patch.py` on llama.cpp, then builds `maya-vision` from
    `tools/vision` with `-DSTRATA_VISION_HIP=ON -DSTRATA_VISION_HIP_TARGETS=gfx1100 -DAMDGPU_TARGETS=gfx1100
    -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_EXE_LINKER_FLAGS=-no-pie -DSTRATA_PORTABLE=ON`.
  - **The ggml ISA fix (`-DSTRATA_PORTABLE=ON`)**: The first CI image (`sha256:e2c3c53b5a7b…`) died with **exit 132
    (SIGILL)** in `ggml_cpu_init` on our AVX2-only i7-13700K because static `ggml-cpu` baked in the GitHub Actions Xeon
    runner's AVX-512 ISA. Passing `-DSTRATA_PORTABLE=ON` sets `GGML_NATIVE=OFF` + `GGML_AVX/AVX2/FMA/F16C/BMI2=ON`
    (asserted in both `CMakeCache.txt` files at build time).
  - **Stamps & runtime**: `deploy/maya/stamps.py` writes `build-hip/MAYA-BUILD.json` (`src=d9009cf853f7c0fa`) and
    `build-vision/MAYA-BUILD.json` (`src=4bc493dc7f779c19`) against `/opt/rocm-7.2.1` so `maya.py` skips recompilation;
    `BUILD_ID` records the git commit; `.venv` pre-installs `setup.PY_PACKAGES` + vendored `textual` wheels. Local image
    `llm-ops-maya:test` (`sha256:58e70296…`, 30.12 GB uncompressed) built in ~8 min.
- **`/entrypoint.sh`**: With no args, serves the newest `/data/config/maya-*-hip.json` (`--backend hip --config … --host
  0.0.0.0 --port $PORT`). Supports `SETUP=1` (guided setup), `PACK_BUILD=1` (self-provisions `/data/vision` and
  `/data/pack/maya-ds4` from read-only `/data/model`), `CONFIG_SRC` (copies repo config JSONs into `/data/config` at
  start, logging any diff), and `DS4_VISION_TOKENS` / `DS4_VISION_MIN_PIXELS` (rewrites `vision.max_image_tokens` in
  `ds4.txt`, clamped at 1140).

## DeepSeek pack, `v1.1.52` A/B vs Strata, and `v1.1.53` parity bump

- **Pack & layout (`/mnt/unenc-xfs/maya/dsv41f1017-iq2xxs/`)**: `tools/ds4/pack_gguf.py` packs the ISTA-DASLab 6-shard
  2.00 bpw (`deepseek41`) set in place and `tools/ds4/vision_pack.py --mmproj` builds `/data/vision/vision.bin` (~0.97
  GB, 266 tensors) from `mmproj-BF16.gguf` (`9a1ef7c4…`). Because `tensors.txt` records `../vision/vision.bin` relative
  to `model_dir`, the read-only shard directory mounts at `/data/model` inside writable `/data`. `DS4_VISION_TOKENS=1140`
  raises the picture cap from 1024 (1680×966 on 4K) to the tower's 10,240-patch softmax limit (1764×1008 on 4K).
- **Testing & speed A/B (`v1.1.52`, 512k, `--ssd-cache 36 --reserve-mib 900`, auto `[0,20]`/`[21,39]` split,
  `run_longctx.py --ctx 524288 --in-pcts 40,85 --out-pct 1 --reps 1`, `cache_n 0`)**: Passed 6/6 functional gates (cold
  load 19–25 s vs Strata's ~41 s, `docker stop` 0.9 s).

  | Arm | pp 40 % | tg 40 % | pp 85 % | tg 85 % | Notes |
  | --- | ---: | ---: | ---: | ---: | --- |
  | Strata 0.1.56 (deployed) | 637.5 | 19.5 | 568.9 | 18.2 | with `STRATA_DS4_HIPK=1 DS4_FAST_ATTN=1` |
  | Maya A0 (`--suffix-draft 3` default) | 656.9 | 23.5 | 628.6 | 23.4 | suffix drafter accepted 41%/13% but slowed decode |
  | Maya A1 (`--suffix-draft 0`) | 696.4 | 24.0 | 621.6 | 24.7 | +20 MiB/card freed (4,231 experts in VRAM: 2,164+2,067) |
  | **Maya A2 (`+ DS4_INDEX_TC=1`, kept)** | **725.1** | **24.8** | **704.7** | **25.4** | **+13.7% / +27% at 40%, +23.9% / +40% at 85% vs Strata** |

  52K knob probes on A1 (3 cold reps): P0 base `632.6 / 21.9`; **P1 `DS4_INDEX_TC=1` (#801) `646.2 / 22.2` (kept)**; P2
  `STRATA_DS4_PREFILL_FUSED=1` `640.3 / 21.8` (rejected, noise); P3 `--ssd-cache off` `629.5 / 20.6` (rejected, −6% tg).
  Switched `deploy/llama-swap.config.yaml` to serve `dsv41f1017-iq2xxs` on Maya (bind-mounting `entrypoint.sh` and
  `configs/`) and retired `strata-dsv41` ([notes/strata.md](strata.md) 20261104;
  [notes/deepseek/v41f1017.md](deepseek/v41f1017.md) 20261104).
- **Maya `1.1.53` parity bump**: Upgraded same day to `v1.1.53` (commit `7c4aa56ed7faea2788705d8e85dd58e892d19f4b`, CI
  run `41844250080`, `ghcr.io/hurui200320/llm-ops-maya:1.1.53-rocm-gfx1100`, digest
  `sha256:c5794fe98f8135ffe9e6e4811269e0d8a97c09265a3d62aafc39ec1ec7505604`). Added #806 BF16 head/shard checks, #688
  GPU-fault restart on next request, and opt-in DS4 knobs (`DS4_AI_RUN`, `MAYA_PROMPT_MEMO`, etc., all default off).
  Ready in ~17 s, 6/6 gates passed.

# 20261105 DeepSeek lane tuning/slots and GLM-5.3-Flash Maya-M-Derisked bring-up

## DeepSeek lane tuning, conversation slots, and per-lane config folders

- **Speed & split tuning (`maya-ds4` 1.1.53, [notes/deepseek/v41f1017.md](deepseek/v41f1017.md) 20261104/20261105)**:
  Tuned `deploy/maya/configs/dsv41f1017-iq2xxs/maya-deepseek-2bpw-hip.json` with `--reserve-mib 1664` (`min free ~=
  reserve - 648 MiB - desktop` ≥ 500 MiB on monitor), `DS4_CHUNK_GFX11=16384` (borrowed 16K prefill chunk),
  `DS4_RING_MB=128`, decode fusions (`DS4_HC_STREAM=1 DS4_HC_MIX_SPLIT=2 DS4_ROUTE_FUSE=1`), and manual split
  `DS4_SPLIT=18,22` (`CUDA0 [0,17]` / `CUDA1 [18,39]+head`), reaching **1,323.25 pp / 26.65 tg at 40% and 1,169.20 pp /
  27.10 tg at 85%** of 512k (4,559 VRAM experts, min free 598 / 1,015 MiB).
- **Conversation slots (`MAYA_DS4_SLOT_KEEP=1 MAYA_DS4_SLOT_CAP=8 MAYA_DS4_SLOT_RAM_GB=6`)**: Holds up to 8 compressed
  KV slots in 6 GB RAM alongside the 36 GB expert store, flushing to `/data/pack/maya-ds4/slots` (16 GB disk cap) on
  `QUIT` (`MAYA_ENGINE_QUIT_S` 50 s, covered by `docker stop -t 120`).
- **Per-lane config folders**: Isolated configs into per-lane subdirectories (`/repo-configs-ds4`, `/repo-configs-glm`,
  `/repo-configs-glm-vanilla`) so `/entrypoint.sh`'s `maya-*-hip.json` glob picks only that lane's config.

## GLM-5.3-Flash Maya-M-Derisked lane bring-up and tuning

- **Layout & one-shot setup (`peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF`, 3 shards = 116.0 GB)**: Because
  `src/core/glm_model.cu` opens shards at `<pack>/../<shard>.gguf` and setup requires writable `--gguf-dir`, each
  production shard is bound read-only **per file** at `/data/<shard>.gguf` beside writable `/data/pack`. Ran `--setup
  --backend hip --gpus 0,1 --yes --no-start --gguf-dir /data --context 524288` (`--gpus 0,1` is mandatory or setup
  writes `"gpu": [0]`), building `/data/pack` (`iq_pack.py --compat-bf16`: 1,257 tensors, 317 native, 0.88 GiB dense,
  NextN MTP `blk.45` 173 MB dense on CUDA1) and fetching `/data/vision` (mmproj 1,075 MB + vocab 9.2 MB).
- **Bring-up & tuning ([notes/glm/5.3-flash.md](glm/5.3-flash.md) 20261105/20261106)**: Initial baseline (`RESERVE_MB=3072`,
  fp16 KV with `--kv-streaming on`, `MAYA_GLM_RAM_GB=36`, disk slots `SLOTS=4/16GB/KEEP=1`, cold load ~33 s / warm ~21 s)
  passed 6/6 gates and slot-restore checks (`cache_n 1730` of 1,744 restored). Tuning adopted reserves `1500/700` →
  `1200/650`, `--prefill 16384` + `MAYA_GLM_PREFILL_MB=6144`, decode take-two kernels (`MAYA_GLM_MLA_ATTN2=h64lvc
  MAYA_GLM_DSA_SCORE2=rl2b4`), and `--kv int8` (+3 GiB `MemAvailable` at speed parity), lifting the 512k ladder from
  `748/740` pp and `16.9/15.8` tg to **897/933 pp and 18.3/19.3 tg** at 40/85% (4,517–4,527 VRAM experts).

# 20261106 Maya 1.1.55 upgrade, GLM 4K picture budget, and vanilla Maya-M lane

## Maya 1.1.55 upgrade and 1.1.53 A/B

Upgraded to `v1.1.55` (commit `c4effee67cb95fb4303720919b075fddf04b4e69`, 52 commits over `v1.1.53`, CI run
`42174794939`, `ghcr.io/hurui200320/llm-ops-maya:1.1.55-rocm-gfx1100`, digest
`sha256:04ac792dac37ebe7768df1a433e4bc3bf644ad149bc3ac09a69cd7b86c759463`, 30,126,717,744 B; llama.cpp/ROCm/hipBLASLt
unchanged).
- **Upstream changes**: New defaults (`0` disables each): `DS4_AI_RUN=1` (prompt-sized indexer slabs),
  `MAYA_DS4_SPEC_HEAD=1`, `MAYA_DS4_DRAFT_SPREAD=1`, `MAYA_DS4_LOAD_INTERLEAVE=1`, `MAYA_PROMPT_MEMO=1`,
  `--decode-cache-every 1024`, `DS4_INDEX_TC=1` default on gfx11, GLM latent KV default `int8`, OpenAI Responses API
  (#726), and version-independent slot identity (DS4 reloaded 13 slots / 2.69 GB, GLM 1 / 1.78 GB).
- **Verification & interleaved 512k A/B (`--in-pcts 40,85 --out-pct 1 --reps 1`, `cache_n 0`, A/B/A/B)**: 6/6 gates
  passed on both lanes (`dsv41f1017-iq2xxs` placed 4,537 VRAM experts vs 4,557 on 1.1.53 due to `DS4_AI_RUN` +
  spec-head buffers).

  | Lane / level | 1.1.53 (r1 / r2) | 1.1.55 (r1 / r2) | Mean Δ | VRAM floors (card0 / card1) & RAM |
  | --- | ---: | ---: | ---: | --- |
  | **DeepSeek** pp / tg 40 % | 1,339.1, 1,331.4 / 27.0, 27.1 | 1,331.0, 1,335.3 / 26.6, 27.0 | −0.2% / −0.9% | 1,004–1,014 / 632–645 MiB; `MemAvailable` 14.4–14.7 GiB |
  | **DeepSeek** pp / tg 85 % | 1,130.8, 1,125.8 / 27.4, 27.8 | 1,176.9, 1,159.9 / 27.5, 27.8 | **+3.6%** / +0.2% | `DS4_AI_RUN` + `MAYA_DS4_SPEC_HEAD` |
  | **GLM (derisked)** pp / tg 40 % | 895.6, 899.8 / 19.6, 19.3 | 900.2, 902.5 / 19.5, 19.2 | +0.4% / −0.5% | 271–274 / 434–453 MiB (reserve 1200/650); `MemAvailable` 13.1 GiB |
  | **GLM (derisked)** pp / tg 85 % | 925.3, 932.1 / 18.8, 19.8 | 937.4, 932.4 / 19.7, 20.2 | +0.7% / +3.4% | inside ±5% spread |

## GLM 4K picture budget (`vision.max_tokens=12288`, `VISION_FOOTPRINT_MB=4711`) and vanilla Maya-M lane

- **GLM 4K picture budget & lend pin**: GLM's tower (`glm5next_mmproj.py`) uses patch-14 × merge-2 (28-px cells = **784
  px/token**), so 3840×2160 needs **10,549 tokens** (137×77 grid) and 4096×2160 needs 11,244 (server auto-fit picks
  4096, downscaling 4K to ~2380×1344). Set `"max_tokens": 12288` under `"vision"`. Because `maya-vision --measure` warms
  up with a fixed 2048×2048 image (5,329 tokens, 3.40 GB lend), a real 4K image (42,196 patches) peaked ~755 MiB higher
  and dropped monitor free VRAM to **35 MiB** at reserve 1200. Fixed in `deploy/maya/entrypoint.sh` via
  `VISION_FOOTPRINT_MB=4711`, which writes `/data/config/vision-memory.json` at start so the engine lends **4.76 GB**
  from the expert pool tail during encode (`210 experts moved to colder slots, 378 dropped`, restored after): at
  reserves **1200/650**, monitor free VRAM is **527 MiB idle, 1,164 MiB during a 4K turn (349 MiB other), and 443 MiB
  on a 9K text prompt**, keeping all 4,569 VRAM experts (`2,265 + 2,304`).
- **Vanilla Maya-M lane (`glm-5.3-flash-iq2s`)**: Provisioned `GLM-5.3-Flash-Maya-M-IQ2_S`
  (`peasantsmith/GLM-5.3-Flash-Maya-GGUF`, `Maya-M`, 3 shards = 116.0 GB, SHA-256s `3f740d1b…` / `285951d2…` /
  `ebf1ce71…`) at `/mnt/unenc-xfs/maya/glm53f-iq2s` with a 1:1 copy of the kept derisked config
  (`deploy/maya/configs/glm53f-iq2s/maya-glm53f-iq2s-hip.json`). Ready in 28 s, 4,569 VRAM experts, 7/7 gates passed
  (4K `prompt_tokens` 10,579), min free 445 / 286 MiB, `MemAvailable` 13.5 GiB.

# 20261107 Maya 1.1.56 upgrade, #861 GLM repack, and 1.1.55 A/B

Upgraded to `v1.1.56` (commit `22fc101f2c109b7e7174c975ff3c2d8e9ce0a867`, 36 commits over `v1.1.55`, CI run
`42229293542`, `ghcr.io/hurui200320/llm-ops-maya:1.1.56-rocm-gfx1100`, digest
`sha256:93723dbc266e3c309281b1f42ed7c4758009f06b0c9612740199f6d305edc7d4`, 30,131,364,031 B; llama.cpp/ROCm/hipBLASLt
unchanged).
- **Upstream changes & #861 GLM repack**:
  - **#861 (`tools/iq_pack.py`)**: Keeps `exp_probs_b.bias` (42 layers × 288 values) in F32 instead of rounding to BF16
    (which left 6–11 distinct values/layer). Rebuilt `/data/pack` on both GLM trees (`compat-bf16.json` 849 → 807
    tensors, index kind 4 → 2, `dense.bin` `943,911,040` → `943,935,232 B`, backup at `<data>/pack-backup-1.1.55/`).
    Because GLM slot identity checks `dense.bin` size, 1.1.56 cleared the stale GLM slots (DS4 reloaded 117 slots /
    7.21 GB unchanged under `engine=0.1.27`).
  - **Other fixes**: #860 two-GPU split drafting restore, #845 fresh random seed when `seed` is omitted (verified on
    GLM), #840 slot-load fallback, #844 DSML tool-loop guard, #809 host-mapped `DS4_AR_P2P` on AMD, and #837 stdlib file
    attachments (`serve/attach.py`).
- **Verification, 1.1.55 vs 1.1.56 A/B, and tuning arms**: All 3 lanes passed 7/7 gates (ready 20 / 31 / 29 s; 4K
  `prompt_tokens` 10,591 on GLM, 1,185 on DS4). Interleaved 512k A/B (`--in-pcts 40,85 --out-pct 1 --reps 1`,
  `cache_n 0`):

  | Lane / level | 1.1.55 (r1 / r2) | 1.1.56 (r1 / r2) | Mean Δ | VRAM floors (card0 / card1) & RAM |
  | --- | ---: | ---: | ---: | --- |
  | **DeepSeek** pp / tg 40 % | 1,361.4, 1,352.8 / 26.9, 27.2 | 1,341.7, 1,350.6 / 26.2, 26.7 | −0.8% / −2.2% | 1,002–1,021 / 626–644 MiB; `MemAvailable` 14 GiB |
  | **DeepSeek** pp / tg 85 % | 1,183.4, 1,159.2 / 27.6, 27.9 | 1,167.8, 1,137.2 / 27.3, 27.7 | −1.6% / −0.9% | 4,555 VRAM experts |
  | **GLM (derisked)** pp / tg 40 % | 900.7, 898.7 / 18.9, 18.9 | 902.8, 903.1 / 19.1, 18.8 | +0.4% / +0.3% | 273–286 / 437–449 MiB; `MemAvailable` 12–13 GiB |
  | **GLM (derisked)** pp / tg 85 % | 940.9, 941.8 / 19.3, 17.9 | 937.0, 935.3 / 18.2, 20.0 | −0.6% / +2.7% | 4,569 VRAM experts, accept 0.700–0.796 |

  - **Tuning arms**: `DS4_CHUNK_GFX11=0` still picks borrowed `prompt chunk 16384` (`1,358.7 / 1,180.7` pp, `26.7 /
    28.1` tg; kept explicit `16384`, [notes/deepseek/v41f1017.md](deepseek/v41f1017.md) 20261107); `MAYA_GLM_P2P=1`
    rejected (`896.8 / 933.5` pp, `19.1 / 17.5` tg, [notes/glm/5.3-flash.md](glm/5.3-flash.md) 20261107).

# 20261110 Maya 1.1.57 upgrade (GLM-5.3 744B bring-up) and 1.1.56 A/B

Upgraded to `v1.1.57` (commit `00655481ce2b0689ab1c2e8d67034dc5701d6511`, 8 commits over `v1.1.56`, CI run
`42659078538`, `ghcr.io/hurui200320/llm-ops-maya:1.1.57-rocm-gfx1100`, digest
`sha256:1372c5d0d33060841cde17951ec3fc4e7708e26e71e425d8128aa941f04a3ee5`, 30,141,963,419 B; llama.cpp/ROCm/hipBLASLt
and `engine=0.1.27` ABI unchanged).
- **Upstream changes**: Adds GLM-5.3 744B (`glm-dsa`, ~256 GB VRAM+RAM, not usable on this host), SIGTERM wait fix
  (`78c450a`), DS4 video-part 400 error, and default-off Flash knobs (`MAYA_GLM_NGRAM_DRAFT`, `MAYA_GLM_LANE_ROW4`,
  `MAYA_GLM_VRAM_EVICT`; prompt cache is hard-gated on `GlmDsa`). No repack needed.
- **Verification, 1.1.56 vs 1.1.57 A/B, and knob screens**: All 3 lanes passed 7/7 gates (ready 27.2 / 36.1 / 32.6 s; 4K
  `prompt_tokens` 10,578 on GLM, 1,145 on DS4; slots reloaded: DS4 131 / 11.22 GB, GLM derisked 4 / 1.66 GB, vanilla 3 /
  4.71 GB). Interleaved 512k A/B (`--in-pcts 40,85 --out-pct 1 --reps 1`, `cache_n 0`; GLM now includes
  `MAYA_GLM_PRESTAGE=160` from 2026-11-07):

  | Lane / level | 1.1.56 (r1 / r2) | 1.1.57 (r1 / r2) | Mean Δ | VRAM floors (other / monitor) & RAM |
  | --- | ---: | ---: | ---: | --- |
  | **DeepSeek** pp / tg 40 % | 1,351.6, 1,351.7 / 27.1, 27.1 | 1,342.3, 1,342.3 / 26.2, 26.8 | −0.7% / −2.2% | 1,003–1,017 / 685–696 MiB; `MemAvailable` 14.5–14.8 GiB |
  | **DeepSeek** pp / tg 85 % | 1,186.7, 1,146.9 / 27.5, 27.1 | 1,165.8, 1,181.7 / 27.3, 27.4 | +0.6% / +0.2% | 10,806–10,820 CPU experts |
  | **GLM (derisked)** pp / tg 40 % | 978.0, 977.5 / 19.0, 19.5 | 978.1, 978.4 / 19.3, 18.4 | +0.05% / −2.1% | 275 / 488 MiB; `MemAvailable` 13.0–13.1 GiB |
  | **GLM (derisked)** pp / tg 85 % | 1,004.5, 996.4 / 17.8, 21.0 | 1,000.5, 999.1 / 19.8, 19.6 | −0.06% / +1.5% | prestage 160, 16K chunks |

  - **GLM 40% knob screens ([notes/glm/5.3-flash.md](glm/5.3-flash.md) 20261110)**: Base `976.3 pp / 19.5 tg`;
    `MAYA_GLM_NGRAM_DRAFT=1` `976.9 / 19.4` (rejected: MTP already hits 99.9% acceptance on verbatim copy and `pick()`
    prefers MTP); `MAYA_GLM_LANE_ROW4=1` `976.8 / 19.4` (rejected: 25.9 ms CPU lane overlaps 24.3 ms disk wait);
    `MAYA_GLM_VRAM_EVICT` skipped.

# 20261111 Maya 1.1.58 upgrade and 1.1.57 A/B

Upgraded to `v1.1.58` (commit `f4067232fa599494122508b55556c4dbaad945ba`, 2 commits over `v1.1.57`, CI run
`42977699783`, `ghcr.io/hurui200320/llm-ops-maya:1.1.58-rocm-gfx1100`, digest
`sha256:f8149c180f0e7e74d3395abe1d3120c6760ea877774f7d493c59bb2724ae0a14`, 30,142,539,960 B; llama.cpp/ROCm/hipBLASLt
and `engine=0.1.27` ABI unchanged).
- **Upstream changes**: `STRATA_GLM_LANE_ROW4` defaults on (bit-exact), server loop guard (#921/#925: user-message
  answers never auto-continue; continued agent answers start a new paragraph), `src/core/glm_ablate.cu` groundwork (off
  without a direction file), and opt-in `MAYA_GLM_PREFILL_STABLE=1` (2–5% pp cost, not deployed). `glm_dsa_pack.py` Q8_0
  attention and DS4 #920/#697 do not touch our packs or 2-GPU split.
- **Verification & interleaved 512k A/B**: All 3 lanes passed 7/7 gates (ready 24.8 / 43.3 / 41.6 s; 4K `prompt_tokens`
  10,577 on GLM, 1,171 on DS4; slots reloaded: DS4 134 / 11.51 GB, GLM 4 + 4). Interleaved 512k A/B (`--in-pcts 40,85
  --out-pct 1 --reps 1`, `cache_n 0`):

  | Lane / level | 1.1.57 (r1 / r2) | 1.1.58 (r1 / r2) | Mean Δ | VRAM floors (other / monitor) & RAM |
  | --- | ---: | ---: | ---: | --- |
  | **DeepSeek** pp / tg 40 % | 1,364.8, 1,241.2 / 27.4, 27.2 | 1,261.5, 1,365.3 / 26.5, 26.5 | +0.8% / −2.9% | 809–821 / 561–566 MiB; `MemAvailable` 12.4–13.2 GiB |
  | **DeepSeek** pp / tg 85 % | 1,158.9, 1,177.4 / 27.4, 27.4 | 1,181.2, 1,183.9 / 27.0, 27.3 | +1.2% / −0.9% | 4,541–4,554 VRAM experts |
  | **GLM (derisked)** pp / tg 40 % | 974.1, 976.8 / 19.0, 19.9 | 977.3, 977.7 / 19.5, 18.7 | +0.2% / −1.8% | 73 / 359 MiB; `MemAvailable` 10.5–11.3 GiB |
  | **GLM (derisked)** pp / tg 85 % | 1,001.9, 1,004.2 / 18.2, 20.4 | 995.8, 999.0 / 19.0, 19.5 | −0.6% / −0.3% | accept 0.763–0.803 |

# 20261112 Maya 1.1.59 (refusal ablation & A/B) and 1.1.60 upgrade

## Maya 1.1.59 upgrade, refusal ablation wiring, and ablation A/B

Upgraded to `v1.1.59` (commit `027992cb1f09829488e650e466233ca351d3d6ef`, 3 commits over `v1.1.58`; built locally on
fedora-tuf with the workflow's exact args after GitHub Actions DNS outages on `security.ubuntu.com`, manifest
`sha256:9c027bf37f0b322a7612d19ece91d6fb25013861c2ad0a54669ead44ecb27806`, stamp `src=95fc0c552f73ea91`, all 7 in-image
gates passed). Upstream renamed internal `STRATA_*` symbols/headers to `MAYA_*` (all 21 deployed env vars keep their
names; `engine=0.1.27` ABI unchanged, DS4 reloaded 134 slots and GLM vanilla 4) and shipped per-request refusal
ablation.

- **Direction files & config wiring (`<tree>/refusal-directions/` = `/data/refusal-directions/`)**:
  - **DeepSeek**: `DeepSeek-V4.1-Flash-refusal-direction.gguf` from `peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF`
    (21,024 B, sha256 `09ab0ac8...`, fitted on Maya-S, layer 25, dim 5120, scale 1.0, think off). (Removed later the
    same day in favor of the static Abiray uncensored lane because think-off direction ablation fails under
    `reasoning_effort: high`; see [notes/deepseek/v41f1017.md](deepseek/v41f1017.md) 20261112 and
    [notes/uncensored.md](uncensored.md).)
  - **GLM vanilla (`glm-5.3-flash-iq2s`)**: `GLM-5.3-Flash-refusal-direction.gguf` from
    `peasantsmith/GLM-5.3-Flash-Maya-GGUF` (16,928 B, sha256 `ecf89232...`, fitted on Maya-S24, layer 31, dim 4096).
    Derisked GLM was retired the same day ([notes/glm/5.3-flash.md](glm/5.3-flash.md) 20261112).
  - **Wiring**: Because `serve` does not derive engine flags from the config's `"ablate"` metadata key at spawn, lane
    configs must carry both `"ablate"` and `--ablate FILE --ablate-default off` in `args`. In
    `deploy/llama-swap.config.yaml`, `filters.setParamsByID` maps `"${MODEL_ID}:ablation"` → `ablate: true`. Plain
    requests stay bit-identical to no direction file; all 7/7 gates passed plain and via `:ablation`.
- **DeepSeek transfer eval (`tools/ablate_fit.py eval --model dsv41f1017-iq2xxs --harmless`, temp 0, think off, 104
  held-out harmful + 104 harmless + 16 coherence prompts)**: Held-out harmful refusals dropped **99.0% → 3.8%**
  (upstream Maya-S: 98% → 1%), harmless false refusals **0.96% → 0.0%**, 0 coherence loops/empty/refusals, length ratio
  **1.02** (`verdict: "ok"`).
- **Ablation speed A/B (`--ablate-default off` vs `on`, 512k, 40/85%, out 1%, `cache_n 0`, A/B/A/B)**:

  | Lane / level | `ablate=off` | `ablate=on` | Mean Δ | Notes & floors |
  | --- | ---: | ---: | ---: | --- |
  | **DS** pp / tg 40 % (2 reps) | 1,315.0, 1,349.6 / 26.60, 26.90 | 1,349.4, 1,287.8 / 26.10, 26.30 | −1.0% / −2.1% | **Free** (within ±3% spread); floors 669–687 / 505–516 MiB |
  | **DS** pp / tg 85 % (2 reps) | 1,183.1, 1,168.2 / 27.40, 27.20 | 1,194.1, 1,181.0 / 27.40, 27.20 | +1.0% / **0.0%** | `MemAvailable` 12.9–13.2 GiB |
  | **GLM** pp 40 % (3 warm reps) | 971.7 (`973.1 / 970.3 / 971.7`) | 968.9 (`967.7 / 969.0 / 969.9`) | −0.3% | After discarding 2 cold-start swap outliers (`MemAvail` ~200 MiB) |
  | **GLM** tg 40 % (3 warm reps) | 18.30 (`18.7 / 17.8 / 18.4`) | 17.57 (`16.8 / 17.7 / 18.2`) | **−4.0%** | Floors 201–239 / 352–372 MiB, `MemAvailable` ~11.7 GiB, accept 0.71–0.80 |
  | **GLM** pp 85 % (3 warm reps) | 985.8 (`990.5 / 990.5 / 976.4`) | 990.9 (`992.0 / 988.7 / 991.9`) | +0.5% | — |
  | **GLM** tg 85 % (3 warm reps) | 18.57 (`18.0 / 17.2 / 20.5`) | 17.67 (`17.0 / 17.6 / 18.4`) | **−4.8%** | Every residual write of every main layer is projected on decode |

## Maya 1.1.60 upgrade (HIP routing-ring sizing)

Upgraded to `v1.1.60` (commit `8b0a449c599fbb370a1dcc0979ae7c84d4310529`, 3 commits over `v1.1.59`, CI run
`43178050443`, `ghcr.io/hurui200320/llm-ops-maya:1.1.60-rocm-gfx1100`, digest
`sha256:7f0dfef0f632f9b756f71c7fe732b444a019e9be5e7ada1ddb38e317965c7f29`, 30,142,642,289 B, stamps
`src=ca1104c6e2bb366d` / vision `src=3485b5bec679d4fa`).
- **Upstream changes & verification**: Sizes the HIP host-mapped routing ring per model (`ring_entries_for` /
  `ring_stride_for`: GLM-5.3-Flash `64 × 4096` floats = 1 MB vs 8.4 MB `256 × 8192` since 1.1.57; glm-dsa `128 × 6144`).
  All 3 active lanes (`dsv41f1017-iq2xxs`, `dsv41f1017-iq2xxs-abiray`, `glm-5.3-flash-iq2s` plain + `:ablation`) passed
  7/7 gates (4K `prompt_tokens` 1,175 on DS, 10,581 on GLM) and preserved slots (DS 135 / 11.52 GB, GLM 4 / 2.00 GB,
  abiray 1 / 0.16 GB). A/B deferred while dev containers occupied the host (measured in the 1.1.61 upgrade below).

# 20261114 Maya 1.1.61 upgrade: DeepSeek prefill +8.6% at 85% on gfx1100

Upgraded to `v1.1.61` (commit `f66a913c628c6672795088a56ad85de0efc827e4`, 4 commits over `v1.1.60`,
`ghcr.io/hurui200320/llm-ops-maya:1.1.61-rocm-gfx1100`, digest
`sha256:83994e79aa49139fe333638f0d9a9665588ac924215450ac4931271129dc896d`, 30,146,959,474 B, stamps
`src=651b9f33b57605a1` / vision `src=19084eccaa9c2513`; GLM/vision sources, pack tools, and `engine=0.1.27` ABI
untouched).
- **Upstream changes (RDNA3 DS4 prefill)**: `set_tc_kernel` adds an RM rows multiplier (decoding an expert weight tile
  once across `RM × 64` rows; default `DS4_IQ_TC_RM=4` on gfx11) and `sparse_attn_wmma_kernel` adds head-group staging
  (`G × 16` heads sharing one staged tile; default `DS4_ATTN_SHARE=4` on gfx11 for chunks ≥ 64 queries). One-GPU-only
  `--fast-prefill` (`DS4_YOCO_CONE=1`) and `MAYA_DS4_LOOKUP_TRUST` (inert with `--suffix-draft 0`) do not apply.
- **Verification & 4-rep interleaved 512k A/B (1.1.60 vs 1.1.61, DeepSeek lane)**: All 3 lanes passed 7/7 gates (4K
  `prompt_tokens` 1,146 on DS, 10,579 on GLM plain + `:ablation`; slots: abiray 215 / 9.26 GB, DS 9 / 1.15 GB after
  16 GB cap rotation, GLM 4 / 1.50 GB). Interleaved 4-rep DS ladder (`tune/ab61/`, `cache_n 0`, VRAM guard before each
  `docker run`):

  | Lane / level | 1.1.60 (r1 / r2 / r3 / r4) | 1.1.61 (r1 / r2 / r3 / r4) | Mean Δ |
  | --- | --- | --- | ---: |
  | **DS** pp 40 % | 1,286.3 / 1,126.2 / 1,262.4 / 1,263.9 (1,234.7) | 1,273.3 / 1,198.2 / 1,375.6 / 1,308.1 (1,288.8) | **+4.4 %** |
  | **DS** tg 40 % | 25.7 / 25.5 / 25.8 / 26.5 (25.88) | 25.2 / 25.5 / 26.2 / 26.3 (25.80) | −0.3 % |
  | **DS** pp 85 % | 1,147.4 / 1,192.5 / 1,144.7 / 1,139.4 (1,156.0) | 1,260.9 / 1,263.0 / 1,235.7 / 1,260.6 (1,255.1) | **+8.6 %** |
  | **DS** tg 85 % | 26.3 / 27.1 / 27.2 / 26.9 (26.88) | 26.7 / 27.1 / 27.2 / 27.2 (27.05) | +0.7 % |

  VRAM floors `838–847 / 609–612 MiB`, `MemAvailable` `12.3–13.0 GiB`, 10,803–10,804 CPU experts. Note: a greedy
  `seed 42` probe before each point showed that DS4 outputs vary boot-to-boot even within the same version (`1.1.60` r3
  vs r4 diverged at byte 150, while `1.1.60` r3 == `1.1.61` r4 was byte-identical), due to boot-varying expert
  admission/association order.

# 20261116 Maya 1.1.64 and 1.1.65 upgrades: DeepSeek and GLM at parity; second GPU1 link drop

## Maya 1.1.64 upgrade and 1.1.61 A/B

Upgraded to `v1.1.64` (commit `d93d1499021b98a39bede20d33368907f1ea4ad8`, 80 commits over `v1.1.61` across
1.1.62–1.1.64, `ghcr.io/hurui200320/llm-ops-maya:1.1.64-rocm-gfx1100`, digest
`sha256:7d371e4f1ac39b433a73531b36f9f18e071360f9b4217aea921ec77b717e26de`, 30,148,744,231 B, stamps
`src=d295b7c3aee9b21f` / vision `src=1c62b3015c165fdf`; `engine=0.1.27` ABI and pack formats unchanged).
- **Upstream changes**: Headline AMD Halo APU + NVIDIA GPU helper mode and `MAYA_GLM_REPEATABLE` / `FIXED_MOVES` /
  `MTP_FIXED` / `CPU_CAL` knobs are gated on `helper.linked()` (inert on 2× RX 7900 XTX). **#956** restores the "3%
  faster or off" rule for DS4's PCIe expert lane on splits (`DS4_STREAM_TIE=keep|off` override); measured neutral on our
  `DS4_SPLIT=18,22` lane below.
- **Verification & interleaved 512k A/B (1.1.61 vs 1.1.64)**: All 3 lanes passed 7/7 gates (+ `:ablation` 9/9 on GLM;
  4K `prompt_tokens` 1,168 on DS, 10,574 on GLM; slots: abiray 217 / 8.7 GB, DS 9 / 0.5 GB, GLM 4 / 0.9 GB). Note: when
  building the GLM tune tree via `rsync --exclude "*.gguf"`, copy `vision/*.gguf` and `refusal-directions/*.gguf` back
  or the entrypoint/engine exits at start; use `GET /unload` on `:8080` to unload llama-swap lanes (`POST` returns 405).

  | Lane / level | 1.1.61 | 1.1.64 | Mean Δ | Floors (other / mon) & notes |
  | --- | --- | --- | ---: | --- |
  | **DS** pp 40 % (4 reps) | 1,383.6 / 1,369.7 / 1,334.2 / 1,385.0 (1,368.1) | 1,367.4 / 1,371.2 / 1,285.1 / 1,363.8 (1,346.9) | −1.6 % | 808–825 / 601–614 MiB |
  | **DS** tg 40 % (4 reps) | 26.40 / 24.80 / 26.30 / 25.70 (25.80) | 25.50 / 25.80 / 25.80 / 26.30 (25.85) | +0.2 % | #956 tie rule neutral here |
  | **DS** pp 85 % (4 reps) | 1,262.9 / 1,265.7 / 1,246.1 / 1,272.1 (1,261.7) | 1,233.2 / 1,219.7 / 1,267.9 / 1,231.5 (1,238.1) | −1.9 % | — |
  | **DS** tg 85 % (4 reps) | 27.10 / 27.40 / 27.20 / 27.20 (27.22) | 26.70 / 27.20 / 27.30 / 27.40 (27.15) | −0.3 % | — |
  | **GLM** pp / tg 40 % (2 reps) | 978.5, 977.6 / 18.50, 17.40 | 959.8, 943.7 / 18.50, 18.80 | −2.7% / +3.9% | 191–194 / 381–383 MiB; bit-probes `9eec69b5...` identical across all 4 points |
  | **GLM** pp / tg 85 % (2 reps) | 963.1, 992.4 / 16.30, 16.30 | 974.9, 965.5 / 18.60, 17.80 | −0.8% / +11.7% | both arms in 16.3–18.8 tg band |

## Maya 1.1.65 upgrade, 1.1.64 A/B, and second GPU1 link drop

Upgraded to `v1.1.65` (commit `c81fab6c9d63cf69b928f92460e0a662b28aa9ec`, 45 commits over `v1.1.64`,
`ghcr.io/hurui200320/llm-ops-maya:1.1.65-rocm-gfx1100`, digest
`sha256:f8a3a10d61f1c6c5fa1d0d2e6fc34cec2b583a5ecb3bc7d36681fcfe61c55823`, 30,144,512,351 B, stamps
`src=a14286b083fafa5b` / vision `src=b94af92b198b2be0`; `engine=0.1.27` ABI unchanged).
- **Upstream changes**: **#899** GLM expert-pool check waits up to 10 s (`MAYA_GLM_POOL_CHECK_WAIT_MS`) for freed VRAM
  to register on Linux HIP before failing with 0 slots/layer; **#827** surfaces engine stops as HTTP 422 `engine_error`
  (or SSE error event) with reason, stage, and token position instead of a silent `length` stop; **#860** logs `glm
  stall:` if a drafting step exceeds `MAYA_GLM_STALL_S=20`.
- **Verification & interleaved 512k A/B (1.1.64 vs 1.1.65)**: All 3 lanes passed 7/7 gates (+ `:ablation` on GLM; 4K
  `prompt_tokens` 1,146 on DS, 10,579 on GLM; slots: abiray 217 / 9.28 GB, DS 8 / 0.39 GB, GLM 4 / 6.06 GB).

  | Lane / level | 1.1.64 (r1 / r2) | 1.1.65 (r1 / r2 / r3) | Mean Δ | Floors (other / mon) & notes |
  | --- | --- | --- | ---: | --- |
  | **DS** pp / tg 40 % | 1,374.2, 1,380.2 / 26.20, 26.10 | 1,384.2, 1,362.5 / 25.80, 26.20 | −0.3% / −0.6% | 810–823 / 611–624 MiB |
  | **DS** pp / tg 85 % | 1,271.3, 1,239.3 / 26.90, 27.10 | 1,265.9, 1,242.1 / 26.40, 27.00 | −0.1% / −1.1% | 3/4 seed-42 probes byte-identical |
  | **GLM** pp / tg 40 % | 973.4, 969.1 / 17.20, 17.40 | 972.9, 971.8, 967.2 / 18.80, 18.00, 17.20 | −0.1% / +4.0% | 186–194 / 389–395 MiB (r3: 109/255 with +250 MiB desktop) |
  | **GLM** pp / tg 85 % | 975.3, 958.9 / 16.80, 16.70 | 978.5, *(r2 lost)*, 985.3 / 18.40, *(r2 lost)*, 18.40 | +1.5% / +9.9% | acceptance-led (0.75–0.77 vs 0.72–0.75); all 5 seed-42 probes `9eec69b5...` |

- **Second GPU1 link drop during `b65-r2` (hardware, not release)**: At 17:52:35 during `b65-r2`'s 85% prefill (~70%
  through), GPU1 (`0000:09:00.0`, PCH x4 riser card) dropped off the bus with the same AER RxErr signature at root port
  `00:1b.4` as 20261109 (`device lost from bus!`, retrained to 2.5 GT/s x4). The #29 watchdog stopped the engine after
  180 s (exit 3) and #827 logged the exact stage (`the prompt: layer 23`) and token position. Force power-off restored
  16 GT/s x4 with 0 errors and `b65-r3` completed cleanly (full record:
  [notes/hardware-failure.md](hardware-failure.md) 20261116).

# 20261119 Maya 1.1.67 upgrade: GLM expert-tier correctness fix, both engines at parity

Upgraded to `v1.1.67` (commit `56b5216ab4464ac41c26b1be32eb63b98de53d0a`, 41 commits over `v1.1.65` across
1.1.66-1.1.67, CI run `44454418707` / job `135325853855`, `ghcr.io/hurui200320/llm-ops-maya:1.1.67-rocm-gfx1100`,
digest `sha256:c5797740e83ca0500db7d409ad3486f14b6f25d3115938ccb702cdd7d94dbba0`, 30,145,736,189 B, stamps
`src=11955178a26fb65b` / vision `src=95d632a5d503389d`; the llama.cpp pin `3cf03257...`, the ROCm base digest
`3db551c4...` and the `engine=0.1.27` session ABI are unchanged - the slots survived).

- **Upstream changes that matter here**: **c38b974** fixes GLM computing an expert with another expert's weights
  (a refill move and a route could land the same expert in two VRAM slots under one key; the clean-up then freed a
  RAM slot the device still pointed at, and a later route read another expert's bytes - silent, no error, frequency
  growing with the share of experts in the RAM tier). Our GLM lane runs `MAYA_GLM_RAM_GB=36` with most experts
  outside VRAM, so it is in the affected class; upstream says answers on affected PCs can differ after the fix
  because they are now correct, speed unchanged. Also here: **#906** (a STOP during a shared start no longer
  disturbs the next turn - the DS lanes keep conversations in slots), **#1004** (a GLM prompt GPU fault names the
  GPU and layer; `MAYA_GLM_PREFILL_CHECK=1` names the stage), **#951** (idle sleeps as settings:
  `MAYA_GLM/DS4_SERVICE_NAP_US` / `_SLEEP_MS`, defaults = the lengths used so far), **#880** (`MAYA_PAR_MAX`,
  inert without `--parallel`), **#923/#972/#973/#927** (setup/web-chat fixes, setup path only) and the EXL3
  `reorder.py` tool (#929; our lanes are GGUF). Inert on this host: the 1.1.66/1.1.67 helper-link decode work
  (AMD APU + NVIDIA, +19 % / +4 %) is gated on `helper.linked()`; the verify-window timed choice is `auto` only
  on compute capability 7.0 and only where every routable expert is in VRAM/helper (unset on gfx1100 still means
  the staggered groups - the "experts outside VRAM ... no timed choice" line never fired in the engine logs); the
  #501 drafter-route profile counts only apply with a drafter (`--suffix-draft 0` on both DS lanes).
- **Verification**: all 3 lanes passed the 7/7 functional gates plus the `:ablation` alias on GLM (6/6 + 4K plain
  and through the alias; 4K `prompt_tokens` 10,579 on GLM - unchanged - and 1,146 on both DS lanes). Slots kept
  across the bump: abiray 217 conversations / 9.28 GB, the main DS lane 81 / 12.37 GB (accumulated since
  20261116; 1 stale conversation removed), GLM 4 / 6.06 GB; live configs clean, no entrypoint diff lines.
- **Tier-table probe**: one 160-token decode on the GLM tune tree with `MAYA_GLM_TIER_VERIFY=1` (128 boundaries a
  card): **all counters zero** - no stale RAM entries, no expert in two VRAM slots, no entry-host splits, no
  routes through a stale entry or a promoted copy (the check is diagnostic only, off by default).
- **Interleaved 512k A/B vs 1.1.65** on the tune trees (`cache_n 0`, `run_longctx --ctx 524288 --in-pcts 40,85
  --out-pct 1 --reps 1`, 2 points an arm each):

  | Lane / level | 1.1.65 (r1 / r2) | 1.1.67 (r1 / r2) | Mean Δ | Floors (other / mon) & notes |
  | --- | ---: | ---: | ---: | --- |
  | **DS** pp 40 % | 1,380.0 / 1,330.3 (1,355.2) | 1,381.6 / 1,344.1 (1,362.9) | +0.6 % | 854-873 / 541-561 MiB; `MemAvailable` 12.9-13.5 GiB, swap >=63.8 GiB free |
  | **DS** tg 40 % | 26.00 / 25.50 (25.75) | 25.40 / 26.00 (25.70) | -0.2 % | — |
  | **DS** pp 85 % | 1,235.5 / 1,258.1 (1,246.8) | 1,263.1 / 1,231.0 (1,247.1) | 0.0 % | — |
  | **DS** tg 85 % | 26.60 / 27.30 (26.95) | 26.90 / 27.50 (27.20) | +0.9 % | — |
  | **GLM** pp 40 % | 967.5 / 967.9 (967.7) | 962.7 / 967.4 (965.1) | -0.3 % | 149-159 / 342-354 MiB; `MemAvailable` 11.6-11.9 GiB, swap >=69.5 GiB free |
  | **GLM** tg 40 % | 17.70 / 17.70 (17.70) | 17.00 / 18.60 (17.80) | +0.6 % | acceptance 0.749-0.750 vs 0.731-0.787 |
  | **GLM** pp 85 % | 970.4 / 976.7 (973.6) | 965.7 / 980.3 (973.0) | -0.1 % | — |
  | **GLM** tg 85 % | 16.80 / 16.30 (16.55) | 16.70 / 18.00 (17.35) | +4.8 % | acceptance-led (0.728 vs 0.752) |

  Both engines at parity; the GLM tg deltas sit inside the lane's 16.3-18.8 t/s band and track draft acceptance,
  pp (the clean compute signal) is unchanged.
- **Answer identity**: all 4 DS bit-probes are byte-identical (`a68ac5c2...`); all 4 GLM probes are byte-identical
  and keep the `9eec69b5...` hash of the 1.1.64/1.1.65 rounds. The tier fix is a race (upstream saw it on 3x V100
  with `MAYA_GLM_REPEATABLE=1`), so a clean probe does not contradict it - the fix is why this RAM-heavy lane is
  trusted after the bump.
- **Deployment**: the three `deploy/llama-swap.config.yaml` lanes now point at `1.1.67-rocm-gfx1100` (commit
  `3f104b5`); llama-swap picked the new image up through its config watcher, and the first starts printed clean
  config installs and the slot restores above.

# 20261121 Maya 1.1.70 upgrade: DS picture-turn start protection, GLM conversation-switch bounce, both engines at parity

Upgraded to `v1.1.70` (commit `b6632a0b11dffd2b197c3bbdcb140bbabbe9fa12`, 188 commits over `v1.1.67` across
1.1.68-1.1.70, CI run `44877650534` / job `136765792643`, `ghcr.io/hurui200320/llm-ops-maya:1.1.70-rocm-gfx1100`,
digest `sha256:3e0329d0e8f8e49a494bb4ad0c8ecc9d2c8556ee4e2ef8f3847c2511ac92998d`, 30,151,233,667 B, stamps
`src=407b05da1087c538` / vision `src=79c45ead63434ff5`; the llama.cpp pin `3cf03257...`, the ROCm base digest
`3db551c4...` and the `engine=0.1.27` session ABI are unchanged - the slots survived). The bump spans three
releases whose headline (v1.1.68's new Hy4-Flash model with its `maya-hy4` engine, ~30k added lines) does not
touch our lanes: `ENGINES=maya,maya-ds4` unchanged, no maya-hy4 build (tracked in [notes/TODO.md](TODO.md)
pending an external score for the model).

- **Integration points verified unchanged before the bump** (v1.1.67..v1.1.70 diff): setup.py's llama.cpp pin,
  `compile_engine_hip`'s cmake line, `PY_PACKAGES`, the gfx1100 hipBLASLt table (62 rows), the session ABI
  (`strata VERSION 0.1.27`) and every deployed `DS4_*` / `MAYA_GLM_*` / `MAYA_DS4_SLOT_*` knob. One new knob:
  `MAYA_DS4_ARENA_THP` (below). No breaking changes.
- **Upstream changes that matter here**:
  - **#1007 (DS)**: prompts with pictures keep restore points - "a client that sends only its newest picture
    had the whole conversation read again for every new picture" (upstream: 284 vs 24,743 tokens on a 24.7K
    chat). The server now asks the engine for restore points on picture prompts too (`marks=`/`shared=`; the
    start log carries "restore points a request asks for: up to 4 a conversation, 2.7 MB of RAM each",
    `MAYA_DS4_RESTORE_POINTS` 4).
  - **#1042 (DS)**: the experts' RAM arena no longer asks for transparent huge pages by default (start 296 s ->
    105 s upstream under a full page cache); `MAYA_DS4_ARENA_THP=1` restores the old behavior. Kept the new
    default (this host runs the DS lanes under documented page-cache/swap pressure).
  - **#1070 (DS, AMD)**: the vision tower's cuBLAS handle/streams are made at the first picture, not at load
    (upstream measured prompts ~25% slower with the tower loaded first on 3x RX 7900 XT). Our start log now
    reads "the model's vision tower runs in the engine (GPU 0: none between pictures, ~332 MiB while one is
    read, lent by the prompt buffers)".
  - **#1006 (DS)**: a short request right after the load waits for the prompt path's GEMM warm-up instead of
    risking the two-GPU warm-up hang (our llama-swap lanes' first request after a cold start is that shape).
  - **#1037 (GLM, AMD)**: conversation switches copy through one pinned 64 MiB buffer kept for the engine's
    life - "a switch no longer stops the answer". Active on our lane: "glm slots: pinned bounce 64 MB (AMD:
    conversation switches free no registered memory)" in the start log.
  - **#935 (DS)**: a failed GGUF expert-cache move is discarded instead of used; **#860** (GLM) names what a
    stalled drafting step waits for; **#1058**: /metrics and request rows count every engine call after a tool
    call or resumed answer; **#1007/3872b40**: an engine abort is no longer reported as "out of RAM"; #1014:
    the tokenizer names token-table corruption.
  - **New knob `cold=1`** (#1012, DS serve): a request line that reads its prompt from the first token, keeping
    nothing from earlier requests - a maya-specific benchmarking control, NOT adopted in bench/ (the runners
    must stay engine-agnostic across llama.cpp/strata/maya).
  - **Expected inert here (verified in code)**: the DS peer all-reduce fixes (#1053, #1059) fire only for
    tensor-parallel groups (`nm > 1`); our lanes are layer-split (`DS4_SPLIT=18,22`, the start log reads "layer
    split across GPUs [0, 1]"). The EXL3-on-AMD kernel work (#968/#1000/#1008, `MAYA_DS4_HC_FUSE` on EXL3) and
    the second-generation IQ2/IQ3 CPU-lane rows don't apply: both DS packs are `format gguf` with GPU-computed
    experts. The GLM Windows-Radeon pin fix (#1035/#924) is Windows-only; the `MAYA_GLM_PREFILL_BLAS` scratch
    sizing (#1047) only fires with that setting on (we don't set it); tools/vision changed MSVC-only.
- **Verification**: all 3 lanes passed the 7/7 functional gates on the new image through llama-swap, plus the
  `:ablation` alias on GLM (6/6 plain + 6/6 through the alias; 4K `prompt_tokens` 10,579 on GLM plain and
  through the alias - unchanged - and 1,146 on both DS lanes). Cold starts: DS 30 s / 30 s, GLM 63 s (the GLM
  wake also re-warmed the 37.9 GB expert RAM tier after two idle days). Live configs clean, no entrypoint diff
  lines; the GLM direction file loads off as before; vision budgets unchanged (1140 DS, 12288 + the 4711 MiB
  footprint pin GLM).
- **Slots across the bump**: the main DS lane 80 conversations / 12.35 GB and GLM 4 / 6.06 GB - kept. The
  abiray lane kept 2 / 0.02 GB and removed **215 as "not this model's or incomplete"**: not a 1.1.70 change -
  the `MAYA_DS4_SLOT_DAYS` 7-day age limit (identical in 1.1.67) expired the lane's 2026-11-12..11-14-era
  conversations (its bring-up and eval days); the 2 survivors' mtimes (11-15, 11-19) sit inside the window and
  their recorded identities match what 1.1.70 computes (byte-compared from the `.meta`). Cache states, not
  data; those conversations re-prefill on return.
- **#1007 on our lanes** (`gate_pics.py`/`gate_pics2.py`, run on the production 1.1.70 lane and a 1.1.67 tune
  arm): the **append path** (turn 2 keeps the old picture and adds a new one) restores fully on **both**
  versions - turn 2 read `cache_n 1161` of 1,694 prompt tokens (wall 37 s -> 6 s) - that is the conversation
  slot, unchanged behavior. The **edited-history path** (turn 2 drops the old picture, as #1007's client) read
  the whole prompt on **both** versions (`cache_n 0`) in a stateless-request harness: the restore points are
  active (the server asks for them; the feature needs the client's conversation continuity to fire, which a
  keyless OpenAI request does not carry). No regression on either path; the fix's gain needs a client that
  keeps the conversation key, which ours do not use.
- **Interleaved 512k A/B vs 1.1.67** on the tune trees (`cache_n 0`, `run_longctx --ctx 524288 --in-pcts 40,85
  --out-pct 1 --reps 1`, 2 points an arm each, `ab70_run.sh`/`glm70_run.sh` from the 1.1.67 templates):

  | Lane / level | 1.1.67 (r1 / r2) | 1.1.70 (r1 / r2) | Mean Δ | Floors (card0 / card1) & notes |
  | --- | ---: | ---: | ---: | --- |
  | **DS** pp 40 % | 1,362.5 / 1,352.1 (1,357.3) | 1,339.5 / 1,326.4 (1,333.0) | -1.8 % | 819-843 / 868-882 MiB; `MemAvailable` 11.2-11.7 GiB, swap >=62.3 GiB free; the delta sits inside the lane's historical run-to-run spread (1,285-1,385 across recent rounds; the 1.1.61->1.1.64 A/B called -1.6 % parity), and no release change touches the DS GGUF prompt path |
  | **DS** tg 40 % | 26.00 / 25.30 (25.65) | 25.50 / 26.10 (25.80) | +0.6 % | — |
  | **DS** pp 85 % | 1,262.7 / 1,255.2 (1,259.0) | 1,236.5 / 1,263.7 (1,250.1) | -0.7 % | — |
  | **DS** tg 85 % | 27.30 / 27.20 (27.25) | 27.20 / 27.30 (27.25) | 0.0 % | — |
  | **GLM** pp 40 % | 966.0 / 965.1 (965.6) | 966.3 / 967.2 (966.8) | +0.1 % | 159-165 / 650-673 MiB; `MemAvailable` 9.5-9.9 GiB, swap >=65.1 GiB free |
  | **GLM** tg 40 % | 17.30 / 18.70 (18.00) | 19.10 / 17.10 (18.10) | +0.6 % | acceptance 0.78-0.79 both arms |
  | **GLM** pp 85 % | 958.1 / 961.9 (960.0) | 961.1 / 973.1 (967.1) | +0.7 % | — |
  | **GLM** tg 85 % | 18.70 / 18.00 (18.35) | 19.30 / 16.50 (17.90) | -2.5 % | acceptance-led (0.794/0.756 vs 0.752/0.731); all points inside the lane's 16.3-18.8 t/s band |

  Both engines at parity: pp (the clean compute signal) is unchanged on GLM and inside the spread on DS; tg
  tracks draft acceptance as on every recent round.
- **Answer identity**: DS - the 1.1.70 arms are byte-identical to each other and to the canonical
  `a68ac5c2...` hash of the 1.1.64/1.1.65/1.1.67 rounds (3 of 4 points; the a67-r2 probe diverged
  `7362338b...`, the lane's documented boot-to-boot variance). GLM - the two arms of each interleaved round
  are byte-identical to each other (r1 `9fa82f8a...` on both, r2 `9eec69b5...` on both; the latter is the
  canonical hash of the previous rounds, reproduced on 1.1.70), with round-to-round boot variance as
  documented since 1.1.61 - no version effect.
- **Deployment**: the three `deploy/llama-swap.config.yaml` lanes point at `1.1.70-rocm-gfx1100` (commit
  `5fdc4bf`); llama-swap picked the new image up through its config watcher, and the first starts printed
  clean config installs and the slot states above. The lanes are left unloaded (lazy start on demand).
