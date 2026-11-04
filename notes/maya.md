# 20261104 Project Maya on gfx1100: image, DeepSeek setup, and the strata -> maya switch

Project Maya (upstream [`mw00/project-maya`](https://github.com/mw00/project-maya)) is a Strata sibling: a Python
installer/dashboard (`maya.py`) that compiles its own C++ engine, plus its own quants of GLM-5.3-Flash (321 B),
DeepSeek-V4.1-Flash-1017 (748 B) and MiniMax-M3.1-Flash (428 B). It speaks OpenAI- and Anthropic-compatible APIs,
reads pictures, and places experts in VRAM / pinned RAM / NVMe by measured use. It targets the 2× RX 7900 XTX pair
(`gfx1100`, 24 GB each; GPU1 behind the Z790 PCH at x4; 62 GiB host RAM) through its experimental HIP backend
(`docs/AMD_MAYA.md` upstream).

Upstream ships no Docker support: `maya.py` compiles the engines on the user's machine at first start and downloads
the model there. Our image does that compile once, the way the installer does it, and ships the binaries plus the
stamps the installer checks, so a first start only downloads/builds the model pack and serves. This entry is the
consolidated setup and test record; the DeepSeek lane now runs Maya because it is faster than the tuned
`strata-dsv41` lane, and the strata side was retired the same day (last section).

## Pins

| What | Value |
| --- | --- |
| Maya | tag `v1.1.52` (`mw00/project-maya`), commit `3c935e411cd6625a9af0f292dd0af52598e866d8` |
| llama.cpp | `3cf03257f219afbe7334045ff7c6a06ac68c627d` (gated against `setup.py` by the workflow) |
| ROCm base | `docker.io/rocm/dev-ubuntu-24.04:7.2.1-complete` |
| Arch | `gfx1100` only |
| hipBLASLt table | `tools/hip/gfx1100-glm-hipblaslt-100202.txt` |

## Setup: the image

### What the installer requires on Linux AMD (from `maya.py`)

`check_hip_pc()` runs at **every start**, not only at setup, and requires: `hipcc` ≥ 7.0 at `<root>/bin/hipcc`,
ROCm's clang (`<root>/llvm/bin`, `<root>/lib/llvm/bin`, a versioned slot, or `<root>/bin`), `libhipblas.so*` under
`<root>/lib|lib64`, a `c++` on PATH, AVX2, and at least one supported GPU in the KFD topology. `<root>` is
`ROCM_PATH`, else `/opt/rocm`, else the newest `/opt/rocm-[0-9]*`, else `/usr`. **A libs-only runtime image cannot
start Maya** - the full SDK (compiler included) must ship.

Engine build (`compile_engine_hip`, Linux branch), reproduced verbatim in `deploy/maya/Dockerfile`:

```
-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF
-DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF
-DSTRATA_PORTABLE=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
-DCMAKE_HIP_COMPILER=<rocm clang++> -DCMAKE_PREFIX_PATH=/opt/rocm
-DSTRATA_GGML_DIR=<llama.cpp>
```

Jobs are `max(1, min(4, cores/2))`. llama.cpp is not vendored: `setup.py` pins the commit above and downloads the
GitHub zip at first setup; `get_llama_cpp()` reuses `third_party/llama.cpp` only when it has `ggml/CMakeLists.txt`
and `gguf-py`. Vision (`compile_vision`, Linux HIP branch) patches llama.cpp with
`tools/vision/glm5next_patch.py` (idempotent), then cmake `tools/vision` with `-DSTRATA_VISION_HIP=ON
-DSTRATA_VISION_HIP_TARGETS=gfx1100 -DAMDGPU_TARGETS=gfx1100 -DCMAKE_HIP_ARCHITECTURES=gfx1100
-DCMAKE_EXE_LINKER_FLAGS=-no-pie`, target `maya-vision` → `build-vision/bin/maya-vision`.

The build writes the installer's rebuild stamps with Maya's own `source_hash` (`deploy/maya/stamps.py`):
`build-hip/MAYA-BUILD.json` (`{backend, src, archs, rocm, gpu_ids, llama, lib_dirs, date, targets}`) and
`build-vision/MAYA-BUILD.json`. `build_step`/`vision_step` reuse the binaries only when these match the source, the
arch list, the **resolved** ROCm root and the target list - so hand-written JSON is not enough; the image must
compute the same hash. hipBLASLt tuning is a path list in `STRATA_HIPBLASLT_TUNING` (Maya reuses Strata's env name):
the setup passes every `tools/hip/<arch>-glm-hipblaslt-*.txt` for the card's arch, and the engine takes, per shape,
the table matching its own hipBLASLt version, else plain hipBLAS. gfx1100 GLM tables ship for `100202`, `100401`,
`100500`.

### ROCm base: dev container, not TheRock wheels

The strata-image workflow builds with TheRock's `gfx110X-dgpu` wheels and ships upstream's package, which bundles
its ROCm libs. Maya has no such package: it links the system ROCm and re-checks it at every start. TheRock wheels
were rejected because that index's newest line (`7.10.0a20251120`) does not reach Maya's ROCm 7.2.1 validation line
or its `100202` gfx1100 table, and the runtime checks would need a hand-assembled SDK anyway. Chosen: AMD's ROCm dev
container following llama.cpp's `.devops/rocm.Dockerfile` pattern - `rocm/dev-ubuntu-24.04:<version>-complete` for
the build stage *and* the runtime stage. ROCm **7.2.1** is the pin: Maya's own RX 7900 validation used system ROCm
7.2.1 / Clang 22, and its hipBLASLt 1.2.2 (`100202`) matches `tools/hip/gfx1100-glm-hipblaslt-100202.txt`. The
`-complete` image is ~7.4 GB compressed and carries the compiler, hipBLAS, hipBLASLt and the device libraries, so
both the build and `check_hip_pc` work.

### Image layout and entrypoint

- Build stage: Ubuntu 24.04 + ROCm 7.2.1 dev (`-complete`) + `build-essential cmake ninja-build python3-venv`; Maya
  source from the build context; llama.cpp at the pin; engines `maya` and `maya-ds4`; `maya-vision`; stamps; `.venv`
  with `setup.PY_PACKAGES` and the vendored `textual` wheels, plus `.strata-pip.json` so the installer does not
  reinstall online.
- Runtime stage: the same ROCm dev image + `libgomp1`, `curl`, `python3-venv`, `build-essential` (the `c++` that
  `check_hip_pc` requires); `/opt/maya` gets the sources (llama.cpp included), the two engine binaries + stamp,
  `build-vision/bin/maya-vision` + stamp, and the venv. CMake object trees stay in the build stage.
- `/entrypoint.sh` (ours; no upstream entrypoint exists): with no arguments it serves the newest
  `/data/config/maya-*-hip.json` (`--backend hip --config … --host 0.0.0.0 --port $PORT`); any arguments go to
  `maya.py` unchanged (`--check`, `--setup`, `--bench`, …). `SETUP=1` runs the first-time setup, which copies the
  config to `/data/config/`. `HEALTHCHECK` on `/health`, `VOLUME /data`, `EXPOSE 8080`.
- `PACK_BUILD=1` is the read-only-model-mount layout used by the DeepSeek lane: build the vision tower and the pack
  from `/data/model` when missing (see below).
- `CONFIG_SRC` is a directory of config JSONs copied into `/data/config` at every start - the repo-carried declared
  state (the Strata `STRATA_SET` equivalent). Before the copy, a live config that differs is summarized in the log
  (`args (-524288; +262144); only in live: expert_profile`), because dashboard edits and maya.py's own schema
  migrations are otherwise silently reset. One maya lane per `CONFIG_SRC` folder (the serve picks the newest JSON).
- `DS4_VISION_TOKENS` / `DS4_VISION_MIN_PIXELS` write the DeepSeek picture budget into every `maya-ds4` pack's
  `ds4.txt` at every start (see below).

### Build verification and the ggml ISA fix

Built 2026-11-04 on fedora-tuf (24 cores / 62 GiB) from `deploy/maya/Dockerfile`:

| What | Value |
| --- | --- |
| Image | `llm-ops-maya:test` (local tag), 30,118,810,113 B uncompressed, id `sha256:58e70296b93bd0652ece5f6a5d025e6780cc0c820b951d5c5eaf385ab843e251` |
| Own layers | ~0.65 GB over the ROCm base: venv 229 MB, `build-hip` 87 MB, `build-vision` 80 MB, llama.cpp source 173 MB |
| Build time (24 cores, 4 jobs) | engines `maya` + `maya-ds4` 173 s, `maya-vision` 195 s, venv 31 s; ~8 min from scratch after the 7.4 GB base pull |
| Binaries | `build-hip/maya` 42,719,832 B, `build-hip/maya-ds4` 48,180,928 B, `build-vision/bin/maya-vision` 83,544,176 B |
| ROCm | `7.2.53211-e1a6bc5663` (hipcc 7.2); hipBLASLt **1.2.2** → version `100202`, the shipped gfx1100 GLM table's version |
| Stamps | engine `src=d9009cf853f7c0fa`, `rocm=/opt/rocm-7.2.1`; vision `src=4bc493dc7f779c19` (`/opt/rocm` is a symlink; the stamp records the resolved root, as `check_hip_pc` resolves it at start) |
| `maya-ds4 --version` | `maya-ds4 1.1.52 (build 3c935e411cd6625a9af0f292dd0af52598e866d8)` (from `BUILD_ID`; a source archive has no `.git`) |

Smoke tests, no GPU needed: `maya --help` rc 0, `maya-ds4 --version`/`--help`, `maya-vision` usage rc 2, `ldd` clean
on all three binaries, venv imports (numpy/jinja2/psutil/jsonschema/yaml/tqdm/requests/PIL/textual), `maya.py
--help`, `c++` present, and the stamp check (`stamps.py verify`). With the GPU (`--device /dev/kfd --device
/dev/dri`, host gids 39/105 on fedora-tuf), `maya.py --check --backend hip --gpu 0` and `--gpus 0,1` both pass: two
`AMD Radeon RX 7900 series (gfx1100), 24 GB` cards, `ROCm: /opt/rocm-7.2.1`, AVX2, "HIP prerequisites found".

Two bring-up fixes: the first runtime tree copied only `maya` (not `maya-ds4`) - the stamp check now catches a
missing engine binary; and `maya-ds4 --version` said "(build unknown)" until the build stage wrote the commit to
`BUILD_ID` (the file `cmake/build_id.cmake` reads for source archives).

**The ggml ISA bug:** the first pushed CI image (`ghcr.io/hurui200320/llm-ops-maya:1.1.52-rocm-gfx1100`, digest
`sha256:e2c3c53b5a7b…`) started, parsed its config, printed its three startup lines, then died **exit 132 (SIGILL)**
in `ggml_cpu_init` at the first `Model::load`. Root cause: `maya-ds4` statically links llama.cpp's ggml-cpu, and
ggml's CMake bakes the **build host's** ISA into it with no runtime variants. GitHub's Actions Xeon has AVX-512; the
AVX2-only runner (i7-13700K, no AVX-512) faults. None of our Strata images link ggml-cpu, so they are unaffected.
Upstream never sees it because `maya.py` compiles on the user's own machine, and the workflow's no-GPU smoke gates
never call `ggml_cpu_init`.

Fix (in `deploy/maya/Dockerfile`): both cmake configure lines take `-DSTRATA_PORTABLE=ON`, upstream's own portable
path (`GGML_NATIVE=OFF` + explicit `GGML_AVX/AVX2/FMA/F16C/BMI2=ON`; `maya.py`'s `check_pc` floor is AVX2, so no
supported host lacks it). A new `RUN` assertion after the builds requires `GGML_NATIVE:BOOL=OFF` and
`GGML_AVX2:BOOL=ON` in both `build-hip/CMakeCache.txt` and `build-vision/CMakeCache.txt`, so a future Maya bump that
changes the mechanism fails the image build instead of a user's first load. Verified by a local rebuild on
fedora-tuf (assertion passes, engine loads the ISTA pack and serves).

## Setup: the DeepSeek pack and data tree

### What Maya accepts

Maya's guided setup validates a `--model-dir` folder only against its own manifest quants (unsloth UD-*, Maya-S/M/XS,
orcarouter); the ISTA-DASLab set is none of them, so the setup would try to download unsloth's. But the pieces the
setup drives are standalone CLIs and take the ISTA set as-is:

- `tools/ds4/pack_gguf.py` is a generic deepseek41 GGUF packer and already carries every quirk of llama.cpp's own
  DeepSeek-V4.1 layout - exactly what ISTA's 2026-10-27 republish is: `general.architecture deepseek41` with `ple.*`
  Engram metadata (`ple_to_engram`), the single `per_layer_token_embd` table in shard 6 (`split_ple`),
  `engram_kv`/`engram_gate` spellings, `exp_probs_b_vl`, 2-D `attn_output_a`, and the ISTA GSQ-RCO shortfall
  workaround (missing DSpark keys, upstream #500).
- The engine reads the ISTA dtype mix natively: routed experts IQ1_M / IQ2_XXS / IQ2_XS (Maya's own low-bit family
  on HIP), dense attn Q5_K / Q8_0, router `ffn_gate_inp` BF16, norms/sinks/hyper-connection F32. Maya's
  `--ssd-cache`, `--reserve-mib`, `--tp`, `--prefill-chunk`, `--threads` are the same flags as Strata's dsv41; the
  server (`maya.py serve/server.py`) reports llama.cpp's own `timings` fields, so `bench/speed/run_longctx.py` works
  unmodified. Maya also has `/tokenize` (Strata's engine does not), so a bench ladder costs a POST instead of an
  uncached calibration prefill (~18 min saved per ladder).
- `tools/ds4/vision_pack.py --mmproj` builds the tower from unsloth's `mmproj-BF16.gguf`, byte-pinned to the same
  sha Strata's lane holds (`9a1ef7c4…`). The dspark Q8 GGUF is not usable by Maya (its drafter comes from the HF
  checkpoint) - parity, since the Strata lane leaves the drafter unloaded too.

### Data tree

One tree per model (the strata pattern), with the shards in place and read-only:

```
/mnt/unenc-xfs/maya/dsv41f1017-iq2xxs/          the lane's data tree (mounted at /data)
  model/      READ-ONLY mount of /mnt/unenc-xfs/models/maya/ISTA-DASLab-dsv41f1017 (the shards, untouched)
  vision/     vision.bin + vision.json (built from the mount's mmproj)
  pack/maya-ds4/   ds4.txt (model_dir /data/model), tensors.txt, engram.bin, tokenizer/, expert profile, caches
  config/     the repo config lands here (CONFIG_SRC)
```

The pack is only an index: `ds4.txt` records `model_dir`, and the engine opens every shard as
`model_dir/<filename>` (`src/ds4/ds4_model.cu` `Pack::load`); the vision tower is a `file` entry in `tensors.txt`
holding the path **relative to model_dir** (`pack_gguf.py` writes `../vision/vision.bin` from `--vision`; the engine
resolves it against `model_dir` at every load - there is no engine flag for it). So the shards mount read-only at
`/data/model`, the tower and pack live in the writable tree as its siblings, and nothing writes beside the
production shards. This is why the shards are mounted **inside** the writable `/data` and not a shared `/models`
mount like strata's: `../vision` must resolve into a writable folder, and `vision.bin` is derived state.

### First start

With `PACK_BUILD=1` and the shard folder at `/data/model`, the first start self-provisions (a restart only stats):

1. **Vision tower** → `/data/vision/`: `tools/ds4/vision_pack.py --dir /data/vision --mmproj
   /data/model/mmproj-*.gguf` (~0.97 GB, self-verified: 266 tensors, every one equal to the mmproj's). Skipped when
   `vision.bin` exists.
2. **Pack** → `/data/pack/maya-ds4/`: `pack_gguf.py --model /data/model --out /data/pack/maya-ds4 --vision
   /data/vision --shards …`, headers only, seconds. The entrypoint passes every `.gguf` in the mount **except
   `mmproj-*.gguf`** (the tower's source, architecture `clip`, which would otherwise be taken as a shard; ISTA's
   sixth shard publishes under the Engram name and is included). Skipped when `ds4.txt` exists.
3. `model_dir` is normalized to `/data/model` in every pack under `/data`, so a host-side rebuilt pack still
   resolves inside the container. The picture budget is applied before the first load, so a fresh pack serves at
   the wanted budget immediately.

The pack took the ISTA set on the first try, no `--shards`/`--engram`/`--without` needed: *"47550 tensors in 7 files
(BF16 1.3 GB, F32 0.2 GB, IQ1_M 59.5 GB, IQ2_XS 23.6 GB, IQ2_XXS 49.0 GB, IQ4_NL 110.6 GB, Q5_K 5.3 GB, Q8_0 0.4
GB), Engram 2 layers x 24 hash columns, the vision tower"*. `ds4.txt` holds the full geometry (vocab 129280, 40
layers, 384 experts, 6/token, head_dim 512, o_groups 8, indexer + source layers, engram layers 1,14, ctx 1048576).
`engram.bin` is 517,980 B, **byte-count-identical to Strata's pack for the same set**.

### Config and vision budget

Maya has no `STRATA_SET` equivalent: `maya.py` reads engine args only from the config JSON, and its start-time
overrides are `--gpu/--gpus/--parallel/--host/--port/--api-key` plus `--env` for engine environment. So the lane
carries its config in the repo: **`deploy/maya/configs/maya-deepseek-2bpw-hip.json`**, mounted read-only at
`/repo-configs` with `CONFIG_SRC=/repo-configs` and copied into `/data/config` at every start. A raw writable
file-mount was rejected: `maya.py` `touch()`es the config at every start and may rewrite it after an upstream bump,
which would dirty the runner's checkout and break its `git pull`. The repo file is the declared state; dashboard
edits last until the next lane restart.

The lane sets `-e DS4_VISION_TOKENS=1140` - the tower's hard maximum (10,240 patches; `kSmT*kSmPer` = 512*20; a
larger value fails with "more than the attention covers"). Maya's pack default 1024 downscales a 4K screenshot to
1680×966; 1140 reaches 1764×1008 (1840×1008 effective for 16:9). There is no engine flag or config key: the pack's
`ds4.txt` key `vision.max_image_tokens` is the only knob, and the entrypoint rewrites it (atomic, only when
different, clamped at 1140, text-only packs skipped) at every start. This survives a repack: `maya.py` re-runs
`pack_gguf.py` when the pack tools' source hash changes (an upstream bump), which resets the key to 1024 - the next
start re-applies the budget. The deployed lane proved the recovery path for real: the pack was reset to 1024 by
hand, the entrypoint re-applied 1140 at that start.

## Testing: DeepSeek-V4.1-Flash 2.00 bpw

### Functional gates

The lane serves the same six-shard ISTA-DASLab set as before. **6/6 gates passed on the first load and again on the
kept config**, text and pictures: `SMOKE-OK` chat; declared `get_weather` → `tool_calls {"city":"Paris"}`;
`reasoning_effort: none` → `42` with no reasoning block; first five primes `2, 3, 5, 7, 11`; `def sum_list` code
block; 420×420 red/blue-halves picture → "The left half is red and the right half is blue." With 1140 tokens, a
3840×2160 red/blue picture read correctly in 14-16 s (prompt 1,166 tokens, ~1,127-token image span, above the old
1024 cap). `docker stop` 0.9 s with VRAM back to desktop levels.

### Speed A/B (512k, `--ssd-cache 36 --reserve-mib 900`)

Method: same-day A0 pairs, the deployed lane through llama-swap vs the direct maya container, the standard ladder
`bench/speed/run_longctx.py --ctx 524288 --in-pcts 40,85 --out-pct 1 --reps 1` (uncached calibration prefills,
`cache_n` 0 verified per rep; 5 s VRAM/RAM/swap sampler). The box's run-to-run spread on this workload is about
±5-10 % pp, ±5 % tg. Maya's load is 19-25 s (the Strata lane's ~41 s).

| arm | pp 40 % | tg 40 % | pp 85 % | tg 85 % |
|---|---:|---:|---:|---:|
| Strata lane 0.1.56, as deployed | 637.5 | 19.5 | 568.9 | 18.2 |
| Maya A0, default `--suffix-draft 3` | 656.9 | 23.5 | 628.6 | 23.4 |
| Maya A1, `--suffix-draft 0` | 696.4 | 24.0 | 621.6 | 24.7 |
| **Maya A2, kept config** | **725.1** | **24.8** | **704.7** | **25.4** |
| **Δ vs strata** | **+13.7 %** | **+27 %** | **+23.9 %** | **+40 %** |

Both sides ran the same ISTA files (hard links), the same `--ssd-cache 36 --reserve-mib 900`, the same 524,288
context and the same auto `[0,20]`/`[21,39]` split. Strata's lane-specific advantages (`STRATA_DS4_HIPK` /
`DS4_FAST_ATTN` prompt kernels, the 1140-token picture budget) had no Maya equivalents, and Maya still won.

The suffix-lookup drafter was net-negative here: at the default 3, Maya accepted 41 % of 59 drafts at 40 % and 13 %
of 24 at 85 %, but decode was **slower** than with it off, and the freed verify buffers gave each card ~20 MiB more
at load (2,163 + 2,068 = 4,231 experts in VRAM, 886/884 MiB free). `--suffix-draft 0` is therefore part of the kept
config; revisit only for heavily quoted/code-edit workloads, which is what the lookup drafter is for.

### Knob probes on the A1 base

Three 52K-token War-and-Peace prefills each, same segments per rep, `cache_n` 0:

| arm | mean pp t/s | Δ vs P0 | tg guardrail | verdict |
|---|---:|---:|---:|---|
| P0 base | 632.6 | — | 21.9 | — |
| P1 `DS4_INDEX_TC=1` (indexer scores on WMMA, gfx1100 opt-in, #801) | 646.2 | +2.0-2.4 % (every rep above every P0 rep) | 22.2 | **kept** |
| P2 `STRATA_DS4_PREFILL_FUSED=1` (fused int8 prompt experts) | 640.3 | +1.0-1.5 % | 21.8 | rejected: inside the ±2 % band |
| P3 `--ssd-cache off` | 629.5 | -0.2-0.7 % | 20.6 | rejected: tg -6 %, frees 33 GiB instead |

P3 doubles as a measurement of the store: with it off, `MemAvailable` jumps 18 → 51 GiB (the store holds ~33 GiB),
pp sits ~-0.5 % and the tg guardrail drops 21.9 → 20.6. The store stays at 36, matching the Strata lane's own ssd36
conclusion. Placement was the standard A1 one on every arm (±4 experts, 882-890 MiB free at load). P1 scales with
prompt length (the indexer's cost grows with context): on the 512k confirmation it gave 725.1 pp at 40 % (+4.1 % over
A1) and 704.7 at 85 % (+13.4 % over A1).

### Placement and kept configuration

At load (A2): CUDA0 layers `[0, 20]` weights 2.87 GB, experts 17.79 GB (2,164 of 8,064), caches 0.48 GB, 886 MiB
free; CUDA1 layers `[21, 39]` + head, weights 2.83 GB, experts 18.74 GB (2,067 of 7,296), caches 0.19 GB, 882 MiB
free; **4,231 experts in VRAM** (±4 across runs); 11,133 on the CPU lane (95.9 GB, in place), 16 threads; RAM store 3,512 slots of
10.2 MB (36.0 GB), filled at load in 4.7 s. The Strata 0.1.56 lane had 4,041 in VRAM, 11,319 on the CPU lane and
3,504 store slots - within a few percent. `--tp 2` is demoted by the engine to the same `[0,20]`/`[21,39]`+head
auto layer split (the IQ2_XXS experts' 256-value blocks split `n_ff 2304` no other way). Maya numbers the cards the
other way: its CUDA0 is the x16 card. It also page-cache pre-reads "the most-routed 47.7 GB read ahead" of the
CPU-lane experts beside the 36 GB store.

Tested (kept) run settings:

```
--pack /data/pack/maya-ds4 --max-context 524288 --tp 2
--ssd-cache 36 --reserve-mib 900 --suffix-draft 0
env DS4_INDEX_TC=1, gpu [0,1], backend hip, vision on
```

The deployed lane uses the same set with `--reserve-mib 1024` in
`deploy/maya/configs/maya-deepseek-2bpw-hip.json`, and the model id `dsv41f1017-iq2xxs`. Vision ran on at Maya's
pack default 1024 during the A/B; the lane now uses 1140.

**Do not repeat:** `--suffix-draft` at its default 3 (decode tax here), `STRATA_DS4_PREFILL_FUSED=1` (+1.2 %, noise),
`--ssd-cache off` (-6 % tg, frees 33 GiB instead), and the CI-built image on AVX-2-only hosts before the
`STRATA_PORTABLE` fix (ggml-cpu SIGILL above).

## Deployment: the lane moves to Maya; strata retires

`deploy/llama-swap.config.yaml`: the DeepSeek lane is now **`dsv41f1017-iq2xxs`** and runs
`ghcr.io/hurui200320/llm-ops-maya:1.1.52-rocm-gfx1100` through the `docker-server-maya-rocm` macro, with
`-v /mnt/unenc-xfs/maya/dsv41f1017-iq2xxs:/data` and the production shard folder read-only at `/data/model`. Same
capabilities as the old lane (`in: [text, image]`, tools, 524,288 ctx, `disableAuto`), `healthCheckTimeout: 600`
(cold start ~25-40 s, stop 0.9 s). The macro bind-mounts the repo's `deploy/maya/entrypoint.sh` over the image's own
and `deploy/maya/configs/` at `/repo-configs`, so entrypoint and engine settings ship with a git pull + lane restart
instead of a ~30 GB image rebuild - the maya analogue of the strata fork mount. The entrypoint copies the repo config
at every start and self-heals a missing pack/tower (`PACK_BUILD=1`) and a repacked picture budget (1140), so a fresh
data tree provisions itself and a deleted pack or tower is rebuilt.

The transition was tested end to end on fedora-tuf with a **fresh data tree**: the tower and pack were rebuilt from
scratch by the entrypoint (pack byte-identical to the hand build), the engine loaded through the read-only mount,
the 6/6 gates passed, and the negative paths behaved (missing shard mount → exit 2 with the fix; `PACK_BUILD` unset
with no pack → exit 2; wrong `CONFIG_SRC` → warning; a hand-edited live config → diff summary, repo state served
again; a plain restart was silent).

The strata side is gone: `.github/workflows/strata-image.yml` no longer builds `strata-dsv41` (the cmake/TheRock/
llama.cpp/`build-dsv41.sh` steps and the binary smoke are removed), `deploy/strata/build-dsv41.sh` is deleted, and
`deploy/strata/entrypoint-local.sh` is stripped to its five Qwen-only changes. Runner-side, the old data tree
`/mnt/unenc-xfs/strata/ISTA-DASLab-dsv41f1017` was deleted, the model folder moved to
`/mnt/unenc-xfs/models/maya/ISTA-DASLab-dsv41f1017/` (a rename; the maya tree's links were the same inodes), and the
unused dspark GGUF + second mmproj were removed (~10.4 GiB freed). A strata revert would now re-run setup from the
GGUFs. Full cleanup record: [notes/strata.md](strata.md) 20261104; lane history:
[notes/deepseek/v41f1017.md](deepseek/v41f1017.md) 20261104.

Post-state: the lane's data tree holds the pack, tower and config (935 MB); the production shards are untouched
(link count 1); llama-swap serves `dsv41f1017-iq2xxs` from the repo config. Not verified here: the other Maya
engines/quant families (GLM-5.3-Flash, MiniMax-M3.1-Flash) and the SETUP path with Maya-managed quants - only the
DeepSeek lane was exercised.
