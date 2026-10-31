# 20261020 Initial Strata deployment (v0.2.0-beta → v0.2.1)

Strata (upstream [`Niko1221/Strata`](https://github.com/Niko1221/Strata)) serves Qwen3.8 Flash-Next
(125B-class MoE, IQ3_S GGUF) on the 2× RX 7900 XTX pair (gfx1100, 24 GB each; GPU1 behind the Z790
PCH at x4; 62 GiB host RAM, Gen4 NVMe, zram + swapfile). Strata's multi-GPU mode is a layer split
with hand-off through pinned host RAM, so RCCL is neither possible nor needed. The container bundles
the engine, both picture readers (CPU and GPU), the RDNA 3 ROCm runtime (`gfx1100`/`1101`/`1102`),
and the Terminal (Python 3.12 + GGUF preparation tools); the host needs only the `amdgpu` kernel
driver.

## Initial `v0.2.0-beta` and `v0.2.1` pins

First brought up on our own CI build (`strata-image.yml`) of `v0.2.0-beta` (tag `v0.2.0-beta`,
commit `3d1fb48090e8f2cefc22d7b5845fb7f5b71f33fa`, engine `0.1.45.3`, asset sha256 `281f6606…`,
pushed digest `sha256:b6e7f1019e560cc9d6d1a294303ccd8baef37c09ef3841b760c2f32c6bb3a85d`, CI run
`39292617432`), then upgraded the same day to upstream's official `0.2.1-rocm-gfx110x` image after a
per-file comparison against our `v0.2.1.1` build showed the engine package byte-identical and the
terminal differing only by a Windows long-path hotfix (`strata-image.yml` retired on 2026-10-20):

| What | Value |
| --- | --- |
| Image | `ghcr.io/niko1221/strata:0.2.1-rocm-gfx110x`, digest `sha256:78551e741f8d47d6edb4b29564cef4e7d33e4f50ab510f3eb1cf0e7adc630fc3` |
| Terminal | commit `c093ac263f8175307e67b7fd7eeeaa2e73ceaded`; `strata 0.2.1 (build 0.1.46; python 3.12.15)` |
| Engine | `strata-engine-0.1.46-linux-x86_64-hip-rx7900`, 241,466,491 B, asset sha256 `20e40a4b426c5da8ba0b5979334eb89538e9c4cd51610d456336153e3d45c764`, manifest `build_id a7bdbfdabc9a+rocm`, variant `rx7900`, 580 files |
| hipBLASLt table | packaged `tuning/gfx1100-hipblaslt-100200.txt`, sha256 `1bb8cd2e232ce1dec5dc0f303e5bd3b32abd9444330a4428a995ad4db8fb97be`, version `100200` |

## Container and launch mechanics discovered during bring-up

- **Engine folder**: `/opt/strata/components/engine/<version>-linux-x86_64-hip-rx7900` (upstream
  drops the `strata-engine-` prefix); always reference it as `component:strata-engine`, never by path.
- **Never pass `KV=`; use `kv=` inside `STRATA_SET`**: `setup.py` accepts only `int8|q4_0|k8v4`,
  while `strata launch` forwards `KV=fp16` straight to setup → argparse error → exit 6, and
  `desc_matches()` repeats the failure on every start. Install with default `int8` and pass `kv=fp16`
  in `STRATA_SET` (lowercase keys become engine flags, replacing `--kv int8` in the engine argv).
- **`PARALLEL` unset is not one request**: the image default fits the card and passes `--batch 3` on
  24 GB. Set `PARALLEL=1` for a single-request lane.
- **Never pass `--slot-save-path`**: it is a `strata serve` option (passing it via `STRATA_SET`
  sends it to the engine and fails with `unknown argument`), and engine `0.1.46+` refuses session
  files with `--layer-split` (`session files do not support --layer-split`). Conversation parking
  (`--conversation-cache-*`) is the working cache mechanism.
- **`STRATA_HIPBLASLT_TABLE` must be a container env var**: since 0.2.1 the engine package bundles
  `tuning/gfx1100-hipblaslt-100200.txt` and `BUILD.json` records `hipblaslt_version: 100200`. On
  gfx1100 the table is opt-in (`-e STRATA_HIPBLASLT_TABLE=on`). The switch is read by the server
  process (`strata_control/hipblaslt_tuning.py` `mode()`) from `os.environ`; putting it inside
  `STRATA_SET` saves it into the preset's `env` and forwards it only to the engine (which reads the
  file path `STRATA_HIPBLASLT_TUNING`, not the switch), silently doing nothing. Verified both ways:
  container env logs `prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)`.
- **`--group-add` takes numeric host gids**: Docker resolves names against the image's `/etc/group`,
  which has no `render` and maps `video` to 44; on fedora-tuf the host gids are `39` (`video`) and
  `105` (`render`).
- **No `--memory` cap**: upstream advises none; the engine plans from host `/proc/meminfo` and the
  entrypoint warns on `memory.max`. The `28 GiB` locked resident budget (`resident-budget-gib=28`)
  is the memory control.
- **Graceful stop**: SIGTERM drains, unloads, and frees VRAM via `strata launch --stop`;
  `--stop-timeout 120` (`docker stop -t 120`) is sized for that.

# 20261021 llama-swap wiring, SC117 lane, and upgrade to 0.2.2

Wired Strata behind host llama-swap (`deploy/llama-swap.config.yaml`, macro
`docker-server-strata-rocm`), added the uncensored SC117 lane alongside the vanilla lane, and
upgraded to upstream's `0.2.2-rocm-gfx110x` image (which added `--no-api-key` in Terminal `0.1.47`,
replacing our 0.2.1 `--network host` + `--access local` workaround with a keyless container on the
default Docker bridge).

## Pins (0.2.2)

| What | Value |
| --- | --- |
| Image | `ghcr.io/niko1221/strata:0.2.2-rocm-gfx110x`, digest `sha256:99ae1768abd51d14ad684f751f42a256ad5f41079adf3ec5da04673c659ebc5b`, 1,842,812,422 B |
| Terminal | tag `v0.2.2`, commit `e6fff4ad12908de74952abe53b2bade45a68aa0f`; `strata 0.2.2 (build 0.1.47; python 3.12.15)` |
| Engine | `strata-engine-0.1.47-linux-x86_64-hip-rx7900`, 242,020,538 B, sha256 `ecaaee56a0fb44313ed246f57655f95857f017a64cb5fb43104b0417c86b21ba`, manifest `build_id 76f90e58436d+rocm`, 581 files |
| hipBLASLt table | packaged `tuning/gfx1100-hipblaslt-100200.txt`, sha256 `1bb8cd2e…`, version `100200` (unchanged) |

## Host and data layout

```
/mnt/unenc-xfs/
  models/strata/
    ISTA-DASLab-Flash-Next/        vanilla IQ3_S pair + pinned mmproj, 84.53 GB
    sc117-Flash-Next-uncensored/   SC117 IQ3_S pair + own mmproj + unused llama.cpp MTP Q8, 88.91 GB
  strata/
    ISTA-DASLab-Flash-Next/        vanilla data tree, 7.9 GB (packs/iq3_s 1.5 GB, mtp/rt 6.5 GB, install/, state/, config/, home/, slots/)
    sc117-Flash-Next-uncensored/   SC117 data tree, 7.9 GB (own pack, copied 6.5 GB MTP, lane state)
  llm-ops/                         host checkout of this repo
    deploy/llama-swap.config.yaml
    deploy/strata/entrypoint-local.sh
```

- GGUFs live once in flat per-lane folders (`/models`), and each lane's data tree is mounted at
  `/data` (bind mounts only, no Docker volumes). Keep each lane's GGUFs and `mmproj` co-located in
  its flat folder: a missing shard fails fast at `check_shards`/`GGUF_DIR`, whereas a missing
  `mmproj` silently downloads ~907 MB into the data tree. Per-file `.sha256` sidecars hold bare
  64-char hex digests.
- Each Qwen lane needs its own `/data` tree because both catalog as tag `iq3_s`. SC117
  (`qwen3.8-flash-next-sc117-iq3s`) is not a drop-in for the vanilla pack: 144 transplanted tensors
  changed 95 tensor types to Q8_0 and moved 1,214/1,223 offsets, so it has a pack built from its own
  shard 1 and reuses a copy of the 6.5 GB `mtp/rt`. Its active `mmproj` is its own 907,542,944 B copy
  (sha256 `2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355`), differing from the
  ISTA pin (`b1a82259…`) only in a 64-byte header (334 tensors identical). The SC117 model folder
  also carries an unused llama.cpp Q8 MTP (4.14 GB, reclaimable).
- `PRESET` is left unset so each lane keeps the image default `docker` preset inside its own tree.

## Keyless bridge and llama-swap integration

- **Entrypoint fork (`deploy/strata/entrypoint-local.sh`, changes 1 & 2)**: upstream's entrypoint
  hardcodes `--access lan` (`0.0.0.0`) and generates an API key. The fork passes `--no-api-key` when
  no key is set and suppresses upstream's false "Strata made one" warning. Each lane's `docker`
  preset is rewritten to `"access": "lan"`, `"no_key": true`, `"api_key": null` (private launch
  config `"api_key": "none"`), and Docker publishes only `-p 127.0.0.1:${PORT}:${PORT}` on the
  default bridge. llama-swap stays `apiKeys: []` and the browser UI works at
  `http://fedora-tuf:8080/upstream/<model>/`.
- **Proxy knobs**: `healthCheckTimeout: 600` (cold start 1–3 min, warm ~30 s; also bounds graceful
  stop so `cmdStop: docker stop -t 120 ${MODEL_ID}` fits), `concurrencyLimit: 20` (queue depth for
  the single engine slot so concurrent requests do not get HTTP 429), `checkEndpoint: /health`, and
  `:npt` alias (`chat_template_kwargs.preserve_thinking: false`).
- **Declared capabilities**: llama-swap picks its prober from `/v1/models`' `owned_by`, which Strata
  omits, so auto-discovery would fall back to the listing-only prober (context + text; no
  tools/image). Strata's `/props` is llama.cpp-shaped (`chat_template_caps`, `modalities`), so
  `capabilities` (`in: [text, image]`, `tools: true`, `context: 262144`, `disableAuto: true`) is
  declared explicitly.
- **0.2.2 verification**: first start dropped the 0.2.1 install record (which named the 0.1.46
  vision reader), re-ran setup once with no repack (~30 s warm), and served keyless on both vanilla
  and SC117 (26.81 GiB resident, `K=22`, 7332 + 6875 expert slots, hipBLASLt enabled, `/health` 200,
  chat, tool call, `:npt`, 640×360 vision with SC117's own mmproj, `/upstream/` UI, `docker stop`
  2.9 s with VRAM freed).

After the 2026-10-22 speed tuning ([`notes/qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md)), both Qwen
lanes adopted:

```
kv=fp16 vram-reserve-mib=1536 vram-reserve-later-mib=768 resident-budget-gib=28
prompt-cache-spread=on conversation-cache-mib=8192 conversation-cache-slots=4
pipeline-windows=2 layer-split=26 trim-stage-weights=on spec-min-p=0.7 STRATA_SCORES_THREAD=1
```

(`layer-split=26` puts layers 0–25 on the x16 card and 26–47 on the x4 card; `pipeline-windows=2`
supersedes `STRATA_SPLIT_OVERLAP` and is incompatible with batch slots; vanilla is fully resident at
262144 ctx with file tier 0.0 GiB, lend 3591/3591, ~31.3 GiB GPU caches, and 22.4 GiB RAM copy.)

# 20261023 Upgrade to 0.2.3 (engine 0.1.48)

Upgraded both Qwen lanes to upstream's `0.2.3-rocm-gfx110x` image and raised
`conversation-cache-slots` from `4` to `8` in `STRATA_SET`.

| What | Value |
| --- | --- |
| Image | `ghcr.io/niko1221/strata:0.2.3-rocm-gfx110x`, digest `sha256:4dd91b54cdf64870b837d00fb031c9b0531d6ea6a9b6fdf72d506912d3029109`, 1,847,098,151 B |
| Terminal | tag `v0.2.3`, commit `5c8dd322528cc84f68f461ad54a506b9da8fa8d4`; `strata 0.2.3 (build 0.1.48; python 3.12.15)` |
| Engine | `strata-engine-0.1.48-linux-x86_64-hip-rx7900`, 242,858,823 B, sha256 `13bcf38c…`, manifest `build_id 6f8800aefe27+rocm`, variant `rx7900`, 581 files (`engine-component.json` sha256 `59a2ac74…`) |
| hipBLASLt table | packaged `tuning/gfx1100-hipblaslt-100200.txt`, sha256 `1bb8cd2e…`, version `100200` (unchanged) |

- First start dropped the 0.2.2 install record (which named 0.1.47 engine/vision paths), re-ran
  setup once, and reused the existing pack and MTP (~40 s warm start, ~30 s of it model load).
- 0.2.3 clamps image tokens by default when `vision.max_tokens` is absent (CPU reader 300, GPU
  reader 1024; #3690). Verified chat, declared tool call, `max_tokens: -1`, and 640×360 vision; spot
  speed on vanilla (60% pp/tg 2077.3/65.95, 85% 2132.5/63.30) was within noise of 0.2.2.
- `0.2.3.1` (Control `0.1.48.1`) was a Windows-only hotfix (`BUILD.json` archs parsing on Windows
  NVIDIA #3810 and Windows HIP engine recognition) with no Docker image and identical 0.1.48 engine
  packages.

# 20261024 Chat template override and upgrade to 0.2.4.2 (own image with strata-dsv41)

## Repo chat template override (entrypoint fork change 3)

Strata has no `--chat-template-file` flag: setup extracts the GGUF template at pack time into
`<pack>/tokenizer/chat_template.jinja`, and the server reads that file on every start. Added
**change 3** to `deploy/strata/entrypoint-local.sh`: when `CHAT_TEMPLATE_PATH` is set, compare it
with every pack's `tokenizer/chat_template.jinja` before `strata launch` and overwrite when they
differ (silent when equal; warns when no pack exists yet). The macro mounts the repo's `templates/`
directory at `/templates:ro` (a directory mount so `fetch-templates.sh`'s atomic rename never leaves
a stale single-file bind), and each model entry sets its own `CHAT_TEMPLATE_PATH`.

Verified on fedora-tuf (2026-10-24) with `templates/qwen3.8/Flash-Next-chat-template.jinja`
(byte-identical to HF `Qwen/Qwen3.8-Flash-Next` `main`, 8,952 B): both packs were updated on start,
a deliberately diverged pack was replaced at the next start (`Chat template: replaced
iq3_s/tokenizer/chat_template.jinja …`), and chat, `get_weather{"city":"Paris"}`, and `:npt` all
passed through llama-swap without repacking or re-downloading GGUFs.

## Upgrade to 0.2.4.2 (`strata-image.yml` restored)

The `0.2.4.x` line (`0.2.4` / `0.2.4.1` / `0.2.4.2`, Control `0.1.49.2`) added DeepSeek-V4.1-Flash
support on a separate `strata-dsv41` engine. Restored `.github/workflows/strata-image.yml` for two
reasons: upstream published no Docker image after 0.2.3, and upstream's signed feed ships
`strata-dsv41` only for `cuda13` and `hip-rx9070` — never for RX 7900 (`gfx110x`). Our workflow
builds Terminal `v0.2.4.2` around the signed-feed 0.1.48 Qwen engine package (applying a build-context
patch for nested `tools/dsv41/` files) and compiles `strata-dsv41` for gfx110x via
`deploy/strata/build-dsv41.sh`.

| What | Value |
| --- | --- |
| Image | `ghcr.io/hurui200320/llm-ops-strata:0.2.4.2-rocm-gfx110x`, digest `sha256:c67827ee283e5e90be31e6c01925727256a26e76aa5b6ee15f57762fa29cf3bc`, 1,889,591,459 B on fedora-tuf (`strata-image` run `40100350366`) |
| Terminal | tag `v0.2.4.2`, commit `dfa3f4025c9f1bc4af8cee201b20a2d223061813`; `strata 0.2.4.2 (build 0.1.49.2; python 3.12.15)` |
| Engine | `strata-engine-0.1.48-linux-x86_64-hip-rx7900` from the `v0.2.4` feed (byte-identical to 0.2.3: 242,858,823 B, asset sha256 `13bcf38c…`, `engine-component.json` `59a2ac74…`, `BUILD.json` `7108976a…`, 580 listed files + manifest + `strata-dsv41` = 582 on disk) |
| `strata-dsv41` | gfx110x build from `v0.2.4.2`, `strata-dsv41 0.1.49.2 build dfa3f4025`, sha256 `53af2567…`, placed beside `strata` in the engine folder |
| hipBLASLt table | packaged `tuning/gfx1100-hipblaslt-100200.txt`, sha256 `1bb8cd2e…`, version `100200` |

Verification (2026-10-24): because the 0.1.48 engine folder was unchanged, both Qwen lanes kept
their 0.2.3 install records and went straight to serving with no setup pass (`Strata-data/` stayed at
50 files, `mtp/tensors/verified.json` untouched). Spot speed (60% pp/tg 2100.5/64.05, 85%
2172.4/64.00) was within noise of 0.2.3; `docker stop -t 120` took 3.8 s with VRAM freed.

# 20261025 DeepSeek-V4.1-Flash lane (dsv41) first deployment

Deployed `deepseek-v4.1-flash-2bpw` serving the IST-DASLab **2.00 bpw** (`IQ2_XXS`) set of
DeepSeek-V4.1-Flash (1017B-parameter MoE, 40 layers × 384 experts, top-6) on the image's
`strata-dsv41` engine. Speed curve and tuning arms: [`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md).

## Model files and pins

| File | Size (B) | sha256 (sidecar, matches upstream's pin) |
| --- | ---: | --- |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-0000{1..4}-of-00004.gguf` (shards) | 39,075,714,624 / 39,388,449,504 / 39,246,891,616 / 20,527,837,312 | `d46efb67…` / `cec1b2c3…` / `9d9c2140…` / `9b790a22…` |
| `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (n-gram table; read from NVMe, never held in RAM) | 110,595,290,816 | `eb19f30d…` |
| `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` (DSpark drafter) | 7,976,914,240 | `6ba6ac1b…` |
| `mmproj-BF16.gguf` (vision tower weights; unused in 0.1.49.2, packed from 0.2.7) | 972,208,288 | `9a1ef7c4…` |
| `DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf` (**not on disk**) | 5,780,589 | `cc15d470…` — `--engram-meta` only; setup downloads it once into `Strata-data/models/deepseek-v4.1-flash-shared/` |

All seven present files match upstream `setup.py` pins (138,238,893,056 B across the four shards;
138 GB model + 111 GB table + 8 GB drafter = 257 GB decimal). The Engram table and DSpark drafter are
shared across 1.75/2.00/2.25 bpw sets.

## Engine discovery gap in Docker (entrypoint fork change 4)

Our gfx110x `strata-dsv41` binary sits beside `strata` in the engine package folder (matching
upstream's `DSV41=1` rx9070 package layout). That layout is auto-discovered only on a desktop install
where `ROOT/engine` is the package folder. In the container, `strata launch` runs setup with
`STRATA_CONTROL_INSTALL=1` / `STRATA_INSTALL_DIR=/data/install`, so `dsv41_exe()` checks only
`/data/install/engine-dsv41/` and `/opt/strata/terminal/engine/`, and `dsv41_engine_found()` probes
only `engine-dsv41/` stores — failing with "DeepSeek's engine is not available for this computer yet".

Fix: **entrypoint fork change 4** exports `STRATA_DSV41_EXE` from
`/opt/strata/components/engine/*/strata-dsv41` when unset (inert for Qwen lanes). Verified in-image
with `strata launch ... --plan` (`DeepSeek engine: installed`).

## Memory plan, RAM budget, and launch settings

- `strata launch ... --plan --gpu 0,1`: `--tp 2`, RAM class "tight" on the 62 GiB host, `--dspark auto
  --dspark-conf 0.65`, hot-expert profile copied from `data/dsv41/profile-2.00.bin` to
  `expert-profile.bin`.
- **RAM budget & VRAM reserve**: unset, `--ssd-cache` auto-sizes to `avail - 12 GB` capped at 48 GB,
  leaving too little RAM for dev work. Qwen's `resident-budget-gib` is ignored by DeepSeek; the knob
  is `--ssd-cache` in decimal GB (`N × 1e9`). First deploy used `ssd-cache=40`; after tuning the
  same day ([`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md)), the lane settled on
  `STRATA_SET="ssd-cache=36 reserve-mib=896"` (36 GB store, ~24 GiB host RAM available + reclaimable
  page cache, and ~350 MiB min free VRAM on the monitor card during prefill). Never pass
  `--ssd-cache 0`.
- **Context, KV, parallel**: `CONTEXT=256k` (`--max-context 262144`). `strata-dsv41` has its own
  MLA-style cache format and no `--kv` flag (`kv=fp16` in `STRATA_SET` exits 2). `--parallel` is
  ignored with a warning (single slot). Thinking is on per the template
  (`templates/deepseek/v4.1-flash-1017-chat-template.jinja`); `reasoning_effort: "none"` or
  `chat_template_kwargs.enable_thinking: false` turns it off.

## Deploy record (2026-10-25)

Init took ~6 minutes: one-time SHA-256 pass over ~258 GB at ~0.7 GB/s, 5.8 MB Unsloth metadata shard
download, 14 MB pack (`Strata-data/packs/deepseek-v4.1-flash-ist-das-2.00bpw`), and the 16-token
smoke (the first kernel execution of `strata-dsv41` on gfx1100).

Engine placement (`state/strata/launch/docker.engine.log`, initial 40 GB store run):

    engine started: .../strata-dsv41 --serve --pack ... --max-context 262144 --eos 1 --dspark auto
                    --dspark-conf 0.65 --tp 2 --ssd-cache 40 --layer-split auto
    ds4: tensor parallel 2 -> 1: the experts' 256-value blocks split n_ff 2304 no other way (each GPU its own layers)
    ds4: DSpark not loaded: the routed experts (132 GB) are far more than the GPU holds (50 GB) ...
    ds4: RAM store: up to 3893 slots of 10.3 MB (40.0 GB), policy 0, 8 reader threads, unbuffered reads
    ds4: RAM store: 3637 of the profile's hottest experts read in (28.2 GB) in 4.7 s
    ds4: CUDA0 layers [0, 20]: weights 2.96 GB, experts 18.74 GB (2256 of 8064 in VRAM), caches 0.24 GB, 586 MiB free
    ds4: CUDA1 layers [21, 39] + head: weights 2.91 GB, experts 19.15 GB (2097 of 7296 in VRAM), caches 0.10 GB, 592 MiB free
    ds4: 11007 routed experts on the CPU lane (94.9 GB, in place in the model's files), 16 threads

- **Placement**: `--tp 2` demoted to a layer split (IQ2_XXS 256-value expert blocks cannot split
  across two cards); 4,353 of 15,360 experts (37.9 GB) in VRAM, 11,007 (94.9 GB) on the CPU lane;
  DSpark auto-unloaded.
- **API & timing**: `/health` 200; `/v1/models` `n_ctx 262144`; `get_weather{"city":"Paris"}`
  (`finish_reason: tool_calls`, DSML translated server-side; `/props` reports `supports_tools: false`
  so llama-swap uses declared `tools: true`); `reasoning_effort: "none"` and `max_tokens: -1`
  verified. Cold start 48.9 s, warm restart 43.4 s, `docker stop` 1.8 s with VRAM back to 26/232 MiB.
- **Port assignment discovery**: llama-swap expands `${PORT}` in **sorted model-ID order**, not file
  order — `deepseek-v4.1-flash-2bpw` got `10001` and the Qwen lanes moved to `10008`/`10009`.
- **Cosmetics**: setup plan logs a spurious "needs more disk space" warning (compares free space
  against the full 257 GB even when files exist on disk), and first start logs "The model already
  installed here is somewhere else than the folder you gave" for multi-file families while still
  reusing the install record.

# 20261025 Upgrade to 0.2.5 (engine 0.1.50)

Upgraded the image to Terminal/Control `0.2.5`/`0.1.50` and engine `0.1.50` across both Qwen and
`strata-dsv41`. Since 0.2.5 upstream ships one engine-less installer per OS and no Docker images.

| What | Value |
| --- | --- |
| Image | `ghcr.io/hurui200320/llm-ops-strata:0.2.5-rocm-gfx110x`, digest `sha256:0bf409c527a5dd9f008d9392ada2457660b3ec2162acf68df6f4822e9b4a092d`, 1,900,632,969 B on fedora-tuf (`strata-image` run `40188533610`) |
| Terminal | tag `v0.2.5`, commit `fa663fabc5c2c568d6035b3ae853b38e1762865a`; Terminal/Control `0.2.5`/`0.1.50`; `strata 0.2.5 (build 0.1.50; python 3.12.15)` |
| Engine | `strata-engine-0.1.50-linux-x86_64-hip-rx7900` from the `v0.2.5` feed, 243,072,991 B, asset sha256 `84698dbf…`, manifest `build_id 6270e2d9d6bf+rocm`, variant `rx7900`, 580 listed files (`engine-component.json` `43746aaf…`, `BUILD.json` `4b70ff83…`) |
| `strata-dsv41` | gfx110x build from `v0.2.5`, `strata-dsv41 0.1.50 build fa663fabc`, sha256 `3b6aa77f…` |
| hipBLASLt table | packaged `tuning/gfx1100-hipblaslt-100200.txt`, sha256 `235a26c3…` (873 B), version `100200` (**re-tuned for 0.1.50**; 0.1.48 shipped `1bb8cd2e…`, 845 B) |

- **Rollout**: engine folder changed (`0.1.48` → `0.1.50`), so each lane dropped its 0.2.4.2 record
  and ran setup once with **no repack and no download** (vanilla first request answered in 45.8 s
  including setup, load, and 8 generated tokens).
- **KV recipe**: `0.2.5`'s `setup.py` now accepts `--kv fp16`, but passing `KV=fp16` would re-run
  setup and rewrite the install record. Kept the lanes installed with `int8` and `kv=fp16` in
  `STRATA_SET` (verified `--kv fp16` in engine argv).
- **Engine changes**: Qwen 0.1.50 added CPU-row kernels (AVX-2 on this i7-13700K; no AVX-512) and the
  re-tuned hipBLASLt table (measured neutral in [`notes/qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md)).
  `strata-dsv41` 0.1.50 added `--experts` (defaults to 6 = Full quality on 48 GiB), length-based
  deterministic chunked prefill from 96 tokens with background warm-up, and opt-in gfx11 WMMA prompt
  kernels (`-e STRATA_DS4_HIPK=1`, enabled on the DeepSeek lane after measuring +18.7% pp in
  [`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md)).

# 20261026 Upgrade to 0.2.6 (strata-dsv41 0.1.51)

Upgraded to Terminal `0.2.6` / Control `0.1.51`. Upstream re-shipped the `0.1.50` rx7900 Qwen engine
package byte-identical; only the Terminal and our `strata-dsv41` binary (`0.1.51`) changed.

| What | Value |
| --- | --- |
| Image | `ghcr.io/hurui200320/llm-ops-strata:0.2.6-rocm-gfx110x`, digest `sha256:46f1ac8203bd5c20bda83a817a10215545cd26d352cccd89dedd9da09315e243`, 1,900,649,052 B on fedora-tuf (`strata-image` run `40313031068`) |
| Terminal | tag `v0.2.6`, commit `4b9e05a2a806a3fe24f7b2560d98626c37dade35`; Terminal/Control `0.2.6`/`0.1.51`; `strata 0.2.6 (build 0.1.51; python 3.12.15)` |
| Engine | `strata-engine-0.1.50-linux-x86_64-hip-rx7900` (same bytes, sha256 `84698dbf…`, manifest `43746aaf…`, and folder path as 0.2.5) |
| `strata-dsv41` | gfx110x build from `v0.2.6`, `strata-dsv41 0.1.51 build 4b9e05a2a`, sha256 `98243abd…` |
| hipBLASLt table | packaged `tuning/gfx1100-hipblaslt-100200.txt`, sha256 `235a26c3…` (unchanged from 0.2.5) |

- Because the engine folder stayed `0.1.50-linux-x86_64-hip-rx7900`, all three lanes kept their
  install records and went straight to serving with no setup pass.
- `strata-dsv41` 0.1.51 added the `--mmap-experts auto` copy-all rule (copies all CPU-lane experts
  into RAM when they fit beside 8 GiB), which cannot trigger on this host (~95 GB CPU experts vs
  62 GiB RAM, plus explicit `ssd-cache=36` disables auto).
- Functional smoke passed on all three lanes through llama-swap (`get_weather{"city":"Paris"}`,
  `:npt`, `/upstream`, DeepSeek `36.0 GB` store + `reasoning_effort: "none"`, stops 2.1–2.3 s).

# 20261027 Qwen3.8 Flash-Next vision token budget 1024–8192 and VRAM reserve tweak

Setup writes the Qwen picture reader's token budget into `/data/install/strata-iq3_s.json` under
`"vision"` (`max_tokens`, defaulting to 1024 on GPU / 300 on CPU; #3690) and there is no launch flag
to override it (`strata launch --set` reaches only the engine). Added **entrypoint fork change 5**:
when `VISION_TOKENS` / `VISION_MIN_TOKENS` are set, patch `max_tokens` and `min_tokens` in every run
config's `vision` section before the server starts, which the server then passes to
`strata-vision-hip` as `--max-tokens` / `--min-tokens`.

- Both Qwen lanes set `VISION_TOKENS=8192` and `VISION_MIN_TOKENS=1024` (matching the Qwen3.8 27B
  llama.cpp lanes). The reader sizes pictures at 1 token per 1024 px (patch 16 × merge 2, 32-px
  aligned): 640×480 = 300 (scaled up to `min_tokens` 1024 for Qwen-VL grounding), 1080p ≈ 2025,
  4K ≈ 8100. Pictures exceeding `max + 12.5%` are refused by the reader.
- Also raised `vram-reserve-later-mib` from `768` to `1024` in the Qwen lanes' `STRATA_SET`.
- Verification on SC117: reader warm-up (2048×2048) moved from 1024 tokens / 0.4 s to 4096 tokens /
  3.5 s, and a 3840×2160 (4K) image encoded to 8160 image tokens (details in
  [`notes/qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md)).

# 20261028 Upgrade to 0.2.7 (engine 0.1.52, DeepSeek vision & 512k context)

Upgraded to Terminal `0.2.7` and engine `0.1.52` — the first Qwen engine byte change since `0.1.50`,
plus `strata-dsv41` `0.1.52` with in-engine vision and 512K context support.

## Pins (0.2.7)

| What | Value |
| --- | --- |
| Image | `ghcr.io/hurui200320/llm-ops-strata:0.2.7-rocm-gfx110x`, digest `sha256:b84c7a5949f6394cb83b5074e0422010cfd1ae82983a9a2ba91ffb9d2fbbf0a1`, 1,939,482,478 B on fedora-tuf (CI reports 1,505,264,720 B; `strata-image` run `40543493857`) |
| Terminal | tag `v0.2.7`, commit `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`; Terminal/Control `0.2.7`/`0.1.52`; `strata 0.2.7 (build 0.1.52; python 3.12.15)` |
| Engine | `strata-engine-0.1.52-linux-x86_64-hip-rx7900` from the `v0.2.7` feed, 245,001,814 B, asset sha256 `2ade169f…`, variant `rx7900`, archs gfx1100/1101/1102, 580 listed files (582 on disk); `engine-component.json` sha256 `a7d13988…`, `BUILD.json` `2f9f4176…` (source `f-amd-vision a42cc3d7df70`) |
| `strata-dsv41` | gfx110x build from `v0.2.7`, `strata-dsv41 0.1.52 build 5e8a4cca7`, sha256 `9c65ec3a43b7f3d1132bc696e7aa18bb6322ce34925f2cd865403c92a8e8b6ea` (CI) |
| hipBLASLt table | packaged `tuning/gfx1100-hipblaslt-100200.txt`, sha256 `1bb8cd2e232ce1dec5dc0f303e5bd3b32abd9444330a4428a995ad4db8fb97be` (845 B), version `100200` — **reverted to the 0.1.48 table bytes** (0.1.50's `235a26c3…` 873 B re-tune is not in this package; still built against hipBLASLt 1.2.0 / TheRock `7.10.0a20251120`) |

## Build, entrypoint, and Host-header changes

- **PR #3974 merged upstream**: `v0.2.7` carries our nested-`tools/` Docker context fix upstream, so
  `strata-image.yml` no longer applies a build patch.
- **Entrypoint**: upstream `docker/entrypoint.sh` added `legacy_models_guard` (#3421), carried
  unchanged in `deploy/strata/entrypoint-local.sh` (inert because every lane sets `MODEL_PATH`).
- **DNS-rebinding Host protection (`STRATA_ALLOWED_HOSTS=fedora-tuf`)**: 0.2.7 checks the `Host`
  header on keyless servers and allows only IP literals, `localhost`/`*.localhost`, and configured
  hostnames. Because llama-swap forwards the client's `Host` header, requests to
  `http://fedora-tuf:8080/...` and `/upstream/<model>/` returned `403 Host ... is not allowed` (while
  LAN-IP curls still worked). Added `-e STRATA_ALLOWED_HOSTS=fedora-tuf` to
  `docker-server-strata-rocm`, fixing all three lanes and `bench/speed/run_longctx.py`.

## DeepSeek vision tower and `DS4_VISION_TOKENS=1140` (entrypoint fork change 6)

- **In-engine vision tower**: `strata-dsv41` 0.1.52 carries its own vision tower (no separate reader
  process). Setup packs the tower metadata from `mmproj-BF16.gguf` (`972,208,288 B`, sha256
  `9a1ef7c4…`), keeping the pack ~8.5 MB and reading tensors from the mmproj in place.
- **Preset gotcha (`-e VISION=on`)**: a lane installed under 0.2.6 had `"vision": "off"` in its saved
  preset, causing the first 0.2.7 setup pass to silently reuse the text-only pack. Re-running setup
  with `-e VISION=on` repacked it with `vision.*`/`image_token_id` and rewrote the preset to
  `"vision": "on"`. Also stop any loaded Qwen lane before running the DeepSeek init smoke (otherwise
  the smoke dies with `cudaMalloc 2 MiB on CUDA0 failed`).
- **Picture budget (`DS4_VISION_TOKENS=1140`, fork change 6)**: the server resizes each image to
  multiples of 42 px (patch 14 × merge 3 = 1 prompt token per 42×42 px cell) so its span
  `n_h*(n_w+1)+2` fits `vision.max_image_tokens` in the pack's `ds4.txt` (setup default: 1024).
  Fork change 6 writes `DS4_VISION_TOKENS` into `ds4.txt` at every start. Set to `1140`, the tower's
  hard maximum (10,240-patch full-attention softmax; 1141 refuses to load; costs ~17 MB more tower
  memory than 1024):

  | Picture aspect | 1024 (setup default) | 1140 (this lane) |
  | --- | --- | --- |
  | Square | 1294×1294 (994 tok) | 1378×1378 (1124 tok) |
  | 16:9 | 1756×966 (991 tok) | 1840×1008 (1082 tok) |
  | 1080p | 1722×966 (968 tok) | 1806×1008 (1058 tok) |
  | 4K UHD | 1680×966 (945 tok) | 1764×1008 (1034 tok) |

  (For comparison, Qwen's 8192-token budget at 32×32 px/token covers ~8.4 MP; DeepSeek's 42×42 px
  cells would need ~4,800 tokens for equivalent pixel area, above the 1,140 attention cap.)

## Rollout verification and DeepSeek 512K context

- **Rollout (2026-10-28)**: every lane dropped its 0.2.6 record (engine folder `0.1.50` → `0.1.52`)
  and ran setup once. First requests through llama-swap including setup took 48.9 s (vanilla) and
  47.5 s (SC117), reusing their packs; Qwen lanes logged `hipBLASLt tuning enabled (26 rows, gfx1100,
  version 100200)` and bit-exact staged HC read on CUDA0/1. DeepSeek warm start took 40.2 s with
  `--vision on`, store `36.0 GB`, and 4,050 experts (35.3 GB) in VRAM vs 4,353 on 0.2.6; verified a
  420×420 red/blue-halves picture (`left=red, right=blue`, first 14.0–14.4 s, warm 1.6 s with
  `cached_tokens 245`) and enabled `capabilities.in: [text, image]`.
- **DeepSeek 512K context (`CONTEXT=512k` → `--max-context 524288`)**: tested and promoted to
  production the same day (`capabilities.context: 524288`). The engine's attention cache doubled
  (`0.24 + 0.10 GB` → `0.48 + 0.19 GB` ≈ 1.3 KB/token; a parked 439K-token conversation is 0.42 GB)
  and moved 54 experts from VRAM to the CPU lane (`4,048` → `3,994` in VRAM), measuring
  speed-neutral against 256k (full numbers in [`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md)).

# 20261030 ISTA's new GGUF for deepseek

**ISTA-DASLab republished the sets on 2026-10-27 in llama.cpp's own DeepSeek-V4.1 layout, and
Strata 0.2.7/0.2.7.1 does not read it. The lane keeps the files above; do not convert and do not
re-download from `main`.** What moved on the Hub: five backbone shards per build plus the shared
Engram as shard 6 (`…-00006-of-00006.gguf` is a link to `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`),
the Engram hash constants as `ple.*` / `engram.token_map` metadata instead of integer tensors of
shard 1, one `per_layer_token_embd` table, `engram_wkv` renamed `engram_kv`, `engram_q`/`engram_k`
folded into `engram_gate`, an added `exp_probs_b_vl` (the vision router bias), `attn_output_a` as a
2-D matrix, and the mmproj renamed `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` with
`clip.projector_type` `deepseek4v` → `deepseek41v`. llama.cpp reads it from b11870; the quantized
weights are the same bytes, only the headers, tensor names and the split points move. The lane's own
`mmproj-BF16.gguf` and drafter come from Unsloth and are already in that layout, so only the four
shards and the Engram header are the old ones.

Why the new files are unusable here, checked against the `v0.2.7` and `v0.2.7.1` trees: `setup.py`
pins `DSV41_ISTA_SHARD` as `…-0000{1..4}-of-00004.gguf` with the sizes and SHA-256 above and the
Engram at `eb19f30d…`, and `HF_REVISIONS` holds the repository at
`cb02422c900df489cea5833e8a92171224059b3c` (2026-10-24) — a setup-driven download still gets the
old layout, and a `MODEL_PATH`/`--gguf-dir` folder of `-of-00006` shards fails the shard check (the
pinned `-of-00004` names are missing). `pack_gguf.py` itself knows the new names
(`engram_kv`, `engram_gate`, `per_layer_token_embd`/`ple.*`, `exp_probs_b_vl`, `deepseek41v` — it
needed them for Unsloth's set), but setup's pins and its `--shards` list do not.

Rules for this lane:

- The pack is filename- and offset-bound: `tensors.txt` names `…-00001-of-00004.gguf` … `00004`,
  the Engram, the drafter and the mmproj. Running ISTA's `convert_ds41_upstream.py` over the folder
  writes five shards with different split points and breaks the pack (the Engram itself is safe:
  its header is rewritten in place at the same data start, so its 110 GB of table bytes stay).
- `strata launch` does not notice such a swap — the run config names only `--pack`, and
  `files_present()`/`desc_matches()` check the pack folder and the context/KV/picture choices, not
  the shards. The container starts, `strata-dsv41` fails to open its files, and the recovery setup
  refuses the new names: the lane is down until the old files are put back.
- Any hand download of these sets pins the revision
  (`--revision cb02422c900df489cea5833e8a92171224059b3c`); the current `main` is the layout setup
  rejects.
- Re-check the pins at the next Strata release (`v0.2.7.1`, 2026-10-28, still had the old ones).
  Moving the lane is then: `convert_ds41_upstream.py` on the folder (needs ~138 GB free for the new
  shards, fetches only `chat_template.jinja` + `exp_probs_b_vl.f32`), repack, re-run the smoke and
  the first-start record. Tracked in [notes/TODO.md](TODO.md).

# 20261030 Upgrade to 0.2.8 (engine 0.1.54)

Upgraded to Terminal `0.2.8` / Control `0.1.54` and engine `0.1.54` across both Qwen lanes and
`strata-dsv41`. `v0.2.7.1` (2026-10-28, not deployed) was Mac-DeepSeek and DGX-Spark work only, so the
effective diff for this box is `v0.2.7..v0.2.8`.

## Pins (0.2.8)

| What | Value |
| --- | --- |
| Image | `ghcr.io/hurui200320/llm-ops-strata:0.2.8-rocm-gfx110x`, digest `sha256:476ded1c680e26e888fc84e93eeec97adcf6997be8124655da9340c1d5c1ba29`, 1,969,765,328 B (`strata-image` run `41015223494`) |
| Terminal | tag `v0.2.8`, commit `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`; `strata 0.2.8 (build 0.1.54; python 3.12.15)` |
| Engine | `strata-engine-0.1.54-linux-x86_64-hip-rx7900`, 250,580,785 B, asset sha256 `4d2833cd95a4912e7e7794f32baa7a00a69830a6980d4ac8a28ef57707b12ee1`, manifest sha256 `e70b103f4b0915339fb00fb992a6721047c8bb6c14a5cd073f3e5de6f4705245`, build_id `8c1aea4bca1b+rocm`, `BUILD.json` source `f-amd-vision 8c1aea4bca1b` |
| `strata-dsv41` | gfx110x build from `v0.2.8`, `strata-dsv41 0.1.54 build 870aedd9e`, sha256 `abc048e909b41bf707681c9d2e32fe6c0cb57b762be886c45961c740965c66d0` |
| hipBLASLt table | packaged `tuning/gfx1100-hipblaslt-100200.txt`, sha256 `1bb8cd2e…` (845 B), version `100200` — **unchanged**; `THEROCK_PIN` stays `7.10.0a20251120` |
| Entrypoint | `/opt/strata/entrypoint.sh` sha256 `10fe9c4f…` — byte-identical to 0.2.7, so the fork carries over unchanged (header re-diffed) |

## Engine and server changes that reach us

- **16-row verify/batch windows on HIP** (`STRATA_WINDOW_MAX_T=auto` → 16; CUDA stays 8; no runtime
  switch). The Qwen engine and our dsv41 build both carry them; upstream measured the answers
  bit-exact (10/10 cards), and our `PARALLEL=1` lanes form narrow windows anyway. dsv41 logs
  `prompt path: the chunked pass from 96 tokens, windows of 16 below`.
- **`tool_call_recovery` is on by default** (was opt-in): slightly malformed calls are repaired,
  extra closing think tags dropped, a missing `</function>` recovered; `"tool_call_recovery": false`
  turns it off. No repair events were logged on well-formed smoke calls; agent-loop quality still
  needs a bench/tool A/B (tracked in [notes/TODO.md](TODO.md)).
- **DeepSeek stop ids (#4576/#4137)**: only a one-token `<|im_end|>` is a stop id, so a tokenizer that
  splits it no longer ends a reply at `|`, `>` or `im` (the "Nerazim" bug); a config that names no
  `arch` is served as `deepseek41` when the tokenizer says so.
- **Fixes on our paths**: the layer-split chunked-prompt deadlock (PR #4211, our DeepSeek topology);
  #4474 (the auto expert cache prices its late allocation) and #4518 (PLE lock vs conversation
  cache) on the Qwen side; #4051 (waits up to 120 s for a leftover engine's memory before a start);
  #4209 (lower-case saved model tags resolve); #4188 (the packaged Terminal runs setup without
  `CMakeLists.txt`).
- **`DS4_FAT_ROWS` is CUDA-only**: `fat_ready()` returns false under `STRATA_USE_HIP`
  (`src/dsv41/ds4_model.cu`), and the string is absent from our binary — the planned prompt-speed A/B
  is moot on gfx1100 and was skipped.
- **ISTA DeepSeek pins unchanged**: `setup.py` still pins `cb02422c…` and the four-shard
  `…-0000{1..4}-of-00004.gguf` names/sizes/hashes (byte-identical to the lane's files). No migration,
  no conversion; the TODO re-check is recorded.

## Rollout (2026-10-30)

Each lane dropped its 0.2.7 install record (engine folder `0.1.52` → `0.1.54`), re-ran setup once
(packs reused, nothing downloaded), and served on the first request: vanilla 49.7 s, SC117 52.3 s,
DeepSeek 45.7 s including setup and load. Both Qwen lanes re-made their run settings because the
record named the 0.1.52 readers, so `VISION_TOKENS` applied on the next start
(`Vision budget: strata-iq3_s.json: max_tokens=8192, min_tokens=1024`); the DeepSeek pack kept
`vision.max_image_tokens 1140`.

Placement on 0.1.54:

- **Qwen (vanilla)**: expert caches 8,304 slots (CUDA0, 15.06 GiB) + 7,849 (CUDA1, 15.72 GiB) =
  16,153 resident experts; 22.87 GiB pinned RAM copy; lend 3,580 slots; file tier 0.0 GiB; hipBLASLt
  enabled (26 rows, version 100200); staged HC read checked bit-for-bit on both cards (0.2.7:
  8,452 + 7,849, 22.62 GiB, lend 3,591 — −0.9 %).
- **DeepSeek**: store 3,504 slots (36.0 GB), 27.2 GB pre-filled; CUDA0 2,037/8,064 experts (16.94 GB,
  892 MiB free), CUDA1 1,957/7,296 (17.87 GB, 886 MiB free) — 3,994 in VRAM, 11,366 on the CPU lane
  (98.0 GB), exactly the 0.2.7 512k placement; DSpark auto-unloaded, vision on,
  `--ssd-cache 36 --reserve-mib 896`.

## Verification and speed

- **Functional** (all through llama-swap): declared `get_weather{"city":"Paris"}` on all three lanes
  (2.3 s / 2.2 s / 15.3 s); `:npt` "OK" on both Qwen lanes; 640×360 vision on both Qwen lanes
  ("Left: Red / Right: Blue", "The left half is red and the right half is blue"); DeepSeek "Nerazim"
  exact (1.1 s, the #4576 case) and a 420×420 picture ("red, blue", 8.8 s) with
  `reasoning_effort: "none"`; `/upstream/` UI 200 (23,050 B), LAN `10.233.1.16:8080` and
  `Host: fedora-tuf:8080` both 200; no tool-call repair events on well-formed calls.
- **Stops**: vanilla 2.9 s, SC117 2.6 s, DeepSeek 3.9 s; VRAM back to 118/259 MiB.
- **Qwen longctx spot** (vanilla, `run_longctx.py --ctx 262144 --out-pct 1 --in-pcts 60,85
  --reps 2`): 60 % pp 2,038.8 / tg 65.50 (0.2.7: 2,063.9 / 64.30); 85 % pp 2,122.1 / tg 62.30
  (2,121.1 / 61.60) — prompts within noise, decode +1–2 %. Numbers and the curve context:
  [`qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md).
- **DeepSeek 512k spot** (shipping lane, `--ctx 524288 --in-pcts 40,85 --out-pct 1 --reps 1`):
  40 % pp 544.1 / tg 19.9 and a repeat at 632.4 / 20.2 (0.2.7: 608.1 / 19.3); 85 % pp 550.1 / tg 18.8
  (547.1 / 19.3). The two 40 % samples straddle the old baseline by −10.5 % / +4.0 %, the run-to-run
  spread on this box, so no delta is claimed; 85 % is equal and decode is within noise. Details:
  [`deepseek/v41f1017.md`](deepseek/v41f1017.md).

# 20261030 A/B mechanics: the preset `extra` accumulates `--set` keys

While A/B-testing knobs on the vanilla Qwen lane (arms and results:
[`notes/qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md)), found the reason "reverting" a `STRATA_SET`
key did nothing: `strata launch --set KEY=VALUE` — which the entrypoint fork drives from
`STRATA_SET`, line 452 — merges the key into the lane's run preset,
`<data>/config/strata/launch-presets/docker.json` → `"extra"`, and every start re-derives the
engine argv/env from it. Removing the key from `STRATA_SET` in `deploy/llama-swap.config.yaml` (or
clearing `env` in `state/strata/launch/docker.json`, which is a derived artifact and gets rewritten)
does **not** remove it: the next launch re-adds it. This is the flip side of the 2026-10-21 finding
that `STRATA_HIPBLASLT_TABLE` inside `STRATA_SET` only reaches the engine.
`STRATA_SCORES_THREAD=1` is live this way: it is in the preset `extra`, not only in `STRATA_SET`.

Practical rules for lane experiments:

- Change the arm in `preset["extra"]` (or in `STRATA_SET` *and* `extra` for a keep), then verify the
  running engine's env (`docker exec <lane> .../proc/<pid>/environ`) and argv (engine log's
  `engine started:` line) — the engine log, not the config file, is the truth.
- Revert by writing `extra` back to the exact baseline map: the twelve flags from the 2026-10-22
  tuning plus `STRATA_SCORES_THREAD=1`; `--prefill` is `auto` (not `auto:16384`), and no
  `STRATA_MTP_*`, `STRATA_EXCHANGE_*` or `STRATA_PCIE_*` keys remain.
- Flags with values (e.g. `--adapt-every`) do not persist outside `extra` (`user_args` stays null),
  but env names absolutely do.

Second quirk seen today: the **first** request that starts a stopped lane through llama-swap can
return `502 Bad Gateway` while the proxy races the serving start; the next request succeeds and the
lane is healthy. A bench script that aborts on its warmup must retry (or pre-warm with a retry
loop).

Final state after the sweep: both Qwen lanes are unchanged from the 2026-10-22 tuning; the vanilla
lane was restarted from the cleaned preset and reproduced its landmarks (expert caches 7959 +
7849, 23.50 GiB pinned resident copy, lend 3581/3581, chunk 8192 ring 96, draft layer on the last
card), with chat, `get_weather{"city":"Paris"}`, `:npt` and 640×360 vision verified through
llama-swap.

# 20261031 Upgrade to 0.2.9 (engine 0.1.55) — ISTA's six-shard DeepSeek layout adopted

Upgraded to Terminal/Control `0.2.9`/`0.1.55` and engine `0.1.55` across both Qwen lanes and
`strata-dsv41`, and moved the DeepSeek lane from the pinned four-shard ISTA layout to the six-shard
set the same day the GGUF files were swapped by hand in the lane's folder. `v0.2.9` is upstream's
first release that reads that layout (the `setup.py` comment calls it "0.2.8.1"; there is no
`v0.2.8.1` tag, so the effective diff here is `v0.2.8..v0.2.9`, tag commit
`c62d5e3a15b40ea51c96aae9e8d36928965f53ab`).

## Pins (0.2.9)

| What | Value |
| --- | --- |
| Image | `ghcr.io/hurui200320/llm-ops-strata:0.2.9-rocm-gfx110x`, digest `sha256:4d369491ccd53fa878cc0b871b52789820981a6059c51c533cce956476fb50a2`, 1,530,632,544 B in CI / 1,971,106,262 B on fedora-tuf (`strata-image` run `41200983707`) |
| Terminal | tag `v0.2.9`, commit `c62d5e3a15b4`; `strata 0.2.9 (build 0.1.55; python 3.12.15)` |
| Engine | `strata-engine-0.1.55-linux-x86_64-hip-rx7900`, 250,890,402 B, asset sha256 `5cfedda68ac3957cae946d387c364db4568da55910d7be933369c04b84d2b01c`, manifest sha256 `b3f45acb0ff8e5d03ecd9821ca48998994d2429023693a334136ed5b77a3e199`, build_id `99866a47a2b8+rocm`, 580 files |
| `strata-dsv41` | gfx110x build from `v0.2.9`, `strata-dsv41 0.1.55 build c62d5e3a1`, sha256 `7738aa7f518b43d115efbf71ce44a1214432c3b636ce6dfa22af449819943f77` |
| hipBLASLt table | packaged `tuning/gfx1100-hipblaslt-100200.txt`, sha256 `1bb8cd2e…` (845 B), version `100200` — unchanged; `THEROCK_PIN` stays `7.10.0a20251120` |
| Entrypoint | `/opt/strata/entrypoint.sh` unchanged from 0.2.8 (sha256 `10fe9c4f…`), so the fork carries over (header re-diffed) |

## Engine changes that reach this box

- **Qwen 0.1.55 is functionally neutral on gfx1100.** The release's new Linux defaults (Gated
  DeltaNet tensor-core prompt kernels, the nine parallel-request settings) are NVIDIA-only, and the
  gfx12 chunked kernel starting at 64 tokens is RDNA4-only. The range `b11889..b12020`-era engine
  diff has no gfx11/HIP kernel change (the `RDNA4_M` split covers gfx1170/1171 only).
- **`strata-dsv41` 0.1.55 reads ISTA's re-split layout** (#4348, #4632): `DSV41_ISTA6_FILES` pins
  the five backbone shards of IQ1_M/IQ2_XS/IQ2_XXS by size and SHA-256; `dsv41_layout_of()` picks
  "six" only when those files are present and the four-shard ones are not; `dsv41_pack_layout()`
  reads a pack's `tensors.txt` and rebuilds a pack made from the other layout. Setup's *download*
  default stays the pinned four-shard revision `cb02422c…`; the six-shard set is taken from local
  files (`--gguf-dir`/`--model-path`).
- **Converted Engram accepted**: the six-shard layout rewrites `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`
  in place (same name and 110,595,290,816 B, one tensor instead of two tables). `DSV41_SHA_ALSO`
  accepts the new hash `3dac4c54…` beside the old pin `eb19f30d…`; either file works with either
  layout.
- **Non-events checked**: `DS4_NO_VIEW` (shards dropped after load) is default-on only for a Windows
  HIP build on an APU, so this Linux dGPU lane keeps the in-place reads; the "DeepSeek uses all AMD
  cards" widening (`widen_setups_one_card`) cannot touch the lane because every launch passes
  `--gpu 0,1` (`gpus_by: "person"`); the local `full-rocm-a07421535` llama.cpp image was unrelated.

## GGUF swap (fedora-tuf, 2026-10-31)

The lane folder `/mnt/unenc-xfs/models/strata/ISTA-DASLab-dsv41f1017/` now holds the six-shard
2.00 bpw set, downloaded by hand (same folder, no path change):

| File | Size (B) | sha256 (sidecar; matches `DSV41_ISTA6_FILES`) |
| --- | ---: | --- |
| `…-IQ2_XXS-00001-of-00006.gguf` (metadata shard) | 5,780,704 | `f356f198…` |
| `…-IQ2_XXS-00002-of-00006.gguf` | 39,535,781,184 | `25af3785…` |
| `…-IQ2_XXS-00003-of-00006.gguf` | 39,736,712,000 | `8168d099…` |
| `…-IQ2_XXS-00004-of-00006.gguf` | 39,036,908,384 | `9eb9b4de…` |
| `…-IQ2_XXS-00005-of-00006.gguf` | 19,923,784,832 | `7e35f6b4…` |
| `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (converted in place) | 110,595,290,816 | `3dac4c54…` |
| `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` (kept) | 7,976,914,240 | `6ba6ac1b…` |
| `mmproj-BF16.gguf` (kept; Unsloth, `deepseek41v`) | 972,208,288 | `9a1ef7c4…` |

The four `-of-00004` shards are removed (their presence would have kept the layout at "four"), and
their `.sha256` sidecars with them. The folder also carries the Hub's renamed
`mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` (972,208,096 B) and its `README.md`: setup looks up
`mmproj-BF16.gguf` by exact name (no globs), so the extra file is inert.

## Rollout

- **Build**: `strata-image` run `41200983707` (v0.2.9, 0.1.55 package, dsv41 rebuilt with the
  unchanged `build-dsv41.sh` flags) finished green in ~25 min; the workflow gate re-checked the
  package (580 files, manifest, `hipblaslt_version 100200`) and the in-image smokes passed.
- **First init attempt** (20:49 local) hashed the five new shards and the Engram (`.done` marks
  written beside the files), rebuilt the pack (six-shard `tensors.txt`, fresh `expert-profile.bin`
  from the shipped `profile-2.00.bin`, `engram.bin` 517,980 B), then died in the 16-token smoke:
  the user's opencode had started the vanilla Qwen lane, which held the GPUs — the documented
  `cudaMalloc` failure. The pack and hash marks survived; the install record did not.
- **Second init** (GPU-free) skipped the hashing and packing, re-checked the pack with the engine,
  passed the smoke and served. Record: engine `0.1.55…/strata-dsv41`, `--max-context 524288`,
  `--vision on`, `gpus 0,1`; `ds4.txt` `vision.max_image_tokens 1140`. Placement was exactly the
  0.2.8 512k one: store 3,504 slots (36.0 GB), 27.2 GB pre-filled; CUDA0 2,037/8,064 experts
  (16.94 GB, 892 MiB free), CUDA1 1,957/7,296 (17.87 GB, 886 MiB free); 11,366 CPU-lane experts
  (98.0 GB, in place); DSpark auto-unloaded.
- **DeepSeek functional smoke** (init server, `reasoning_effort: "none"`): "OK"; `Nerazim` exact
  (the #4576 case); `get_weather{"city":"Paris"}` → `tool_calls` with
  `{"city":"Paris"}`; a 64×64 red/blue-halves PNG → "Red, Blue". Init stopped with VRAM freed.

The Qwen lanes' 0.2.9 pass (engine folder `0.1.54` → `0.1.55`, one setup pass each, no downloads,
vision budget re-applied on the next start) and their smoke are recorded below once the config
carrying the new tag is on the runner.
