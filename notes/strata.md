# 20261020 Initial Strata deployment (v0.2.0-beta → v0.2.1)

Strata (upstream [`Niko1221/Strata`](https://github.com/Niko1221/Strata)) serves Qwen3.8 Flash-Next (125B-class MoE,
IQ3_S GGUF) on our 2× RX 7900 XTX pair (`gfx1100`, 24 GB each; GPU1 behind the Z790 PCH at x4; 62 GiB host RAM, Gen4
NVMe, zram + swapfile). Strata's multi-GPU mode is a layer split with hand-off through pinned host RAM (no RCCL). The
container bundles the engine, CPU/GPU picture readers, RDNA 3 ROCm runtime (`gfx1100`/`1101`/`1102`), and Terminal
(Python 3.12 + GGUF prep tools); the host needs only the `amdgpu` kernel driver.

## Initial `v0.2.0-beta` and `v0.2.1` pins

First brought up on our CI build (`strata-image.yml`, run `39292617432`) of `v0.2.0-beta` (commit
`3d1fb48090e8f2cefc22d7b5845fb7f5b71f33fa`, engine `0.1.45.3`, asset sha256 `281f6606…`, image digest
`sha256:b6e7f1019e560cc9d6d1a294303ccd8baef37c09ef3841b760c2f32c6bb3a85d`), then switched the same day to upstream's
official `0.2.1-rocm-gfx110x` image after verifying the engine package was byte-identical to our `v0.2.1.1` build:

- **Image**: `ghcr.io/niko1221/strata:0.2.1-rocm-gfx110x`, digest
  `sha256:78551e741f8d47d6edb4b29564cef4e7d33e4f50ab510f3eb1cf0e7adc630fc3`
- **Terminal & Engine**: commit `c093ac263f8175307e67b7fd7eeeaa2e73ceaded` (`strata 0.2.1`, build `0.1.46`);
  `strata-engine-0.1.46-linux-x86_64-hip-rx7900` (241,466,491 B, asset sha256
  `20e40a4b426c5da8ba0b5979334eb89538e9c4cd51610d456336153e3d45c764`, `build_id a7bdbfdabc9a+rocm`, variant `rx7900`,
  580 files)
- **hipBLASLt table**: packaged `tuning/gfx1100-hipblaslt-100200.txt`, sha256
  `1bb8cd2e232ce1dec5dc0f303e5bd3b32abd9444330a4428a995ad4db8fb97be` (845 B), version `100200`

## Container and launch mechanics discovered during bring-up

- **Engine folder**: `/opt/strata/components/engine/<version>-linux-x86_64-hip-rx7900` (reference as
  `component:strata-engine`, never by path).
- **Never pass `KV=`; use `kv=` inside `STRATA_SET`**: `setup.py` before 0.2.5 accepted only `int8|q4_0|k8v4` (`KV=fp16`
  exited 6 on every start). Install with default `int8` and pass `kv=fp16` in `STRATA_SET` (lowercase keys become engine
  flags, overriding `--kv int8` in engine argv).
- **`PARALLEL` unset is not 1**: image default passes `--batch 3` on 24 GB; set `PARALLEL=1` for a single-request lane.
- **Never pass `--slot-save-path`**: it is a `strata serve` option (fails in `STRATA_SET`), and engine `0.1.46+` refuses
  session files with `--layer-split`. Use `--conversation-cache-*` instead.
- **`STRATA_HIPBLASLT_TABLE=on` must be a container env var**: read by the server (`strata_control/hipblaslt_tuning.py`
  `mode()`) from `os.environ`. Putting it in `STRATA_SET` forwards it only to the engine (which reads the file path
  `STRATA_HIPBLASLT_TUNING`, not the switch) and silently does nothing. Container env logs `prefill gemm: hipBLASLt
  tuning enabled (26 rows, gfx1100, version 100200)`.
- **Host GIDs, memory, and stop**: pass numeric host GIDs (`--group-add 39 --group-add 105` on fedora-tuf; image has no
  `render` and maps `video` to 44). Set no `--memory` cap (engine plans from `/proc/meminfo`; use
  `resident-budget-gib=28`). SIGTERM runs `strata launch --stop`; size `--stop-timeout 120` (`docker stop -t 120`).

# 20261021 llama-swap wiring, SC117 lane, and upgrade to 0.2.2

Wired Strata behind host llama-swap (`deploy/llama-swap.config.yaml`, macro `docker-server-strata-rocm`), added the
uncensored SC117 lane alongside vanilla, and upgraded to `0.2.2-rocm-gfx110x` (Terminal `0.1.47` added `--no-api-key`,
replacing our 0.2.1 `--network host` + `--access local` workaround with a keyless container on the default bridge).

- **Pins (0.2.2)**: `ghcr.io/niko1221/strata:0.2.2-rocm-gfx110x`, digest
  `sha256:99ae1768abd51d14ad684f751f42a256ad5f41079adf3ec5da04673c659ebc5b` (1,842,812,422 B); Terminal tag `v0.2.2`,
  commit `e6fff4ad12908de74952abe53b2bade45a68aa0f` (build `0.1.47`); `strata-engine-0.1.47-linux-x86_64-hip-rx7900`
  (242,020,538 B, sha256 `ecaaee56a0fb44313ed246f57655f95857f017a64cb5fb43104b0417c86b21ba`, `build_id
  76f90e58436d+rocm`, 581 files); hipBLASLt table unchanged (`1bb8cd2e…`, `100200`).
- **Host layout (`/mnt/unenc-xfs/`)**: GGUFs + co-located `mmproj` + `.sha256` sidecars in `/models`
  (`models/strata/ISTA-DASLab-Flash-Next/` 84.53 GB; `models/strata/sc117-Flash-Next-uncensored/` 88.91 GB) and per-lane
  data trees at `/data` (`strata/ISTA-DASLab-Flash-Next/` and `strata/sc117-Flash-Next-uncensored/`, 7.9 GB each:
  `packs/iq3_s` 1.5 GB, `mtp/rt` 6.5 GB, `install/`, `state/`, `config/`, `home/`, `slots/`). Keep `mmproj` co-located
  with the shards (a missing `mmproj` silently downloads ~907 MB into `/data`). SC117 (`qwen3.8-flash-next-sc117-iq3s`)
  needs its own `/data` tree and pack because 144 transplanted tensors changed 95 types to Q8_0 and shifted 1,214/1,223
  offsets; its `mmproj` (907,542,944 B, sha256 `2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355`)
  differs from ISTA's (`b1a82259…`) only in a 64-byte header.
- **Keyless bridge & proxy wiring**: Entrypoint fork (`deploy/strata/entrypoint-local.sh`, **changes 1 & 2**) passes
  `--no-api-key` when `API_KEY` is unset and suppresses the false "Strata made one" warning; Docker binds
  `-p 127.0.0.1:${PORT}:${PORT}` and the UI is reachable at `/upstream/<model>/`. Proxy sets `healthCheckTimeout: 600`,
  `concurrencyLimit: 20`, `checkEndpoint: /health`, `:npt` alias (`preserve_thinking: false`), and explicit
  `capabilities` (`in: [text, image]`, `tools: true`, `context: 262144`, `disableAuto: true` because Strata omits
  `owned_by` on `/v1/models`).
- **Verification & tuned `STRATA_SET`**: 0.2.2 dropped the 0.2.1 install record, re-ran setup without repacking (~30 s
  warm), and passed all smokes on both lanes. After the 2026-10-22 tuning
  ([`notes/qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md)), both lanes adopted:
  `kv=fp16 vram-reserve-mib=1536 vram-reserve-later-mib=768 resident-budget-gib=28 prompt-cache-spread=on
  conversation-cache-mib=8192 conversation-cache-slots=4 pipeline-windows=2 layer-split=26 trim-stage-weights=on
  spec-min-p=0.7 STRATA_SCORES_THREAD=1` (layers 0–25 on x16, 26–47 on x4; file tier 0.0 GiB, lend 3591/3591, ~31.3 GiB
  GPU caches, 22.4 GiB RAM copy).

# 20261023 Upgrade to 0.2.3 (engine 0.1.48)

Upgraded both Qwen lanes to `ghcr.io/niko1221/strata:0.2.3-rocm-gfx110x` (digest
`sha256:4dd91b54cdf64870b837d00fb031c9b0531d6ea6a9b6fdf72d506912d3029109`, 1,847,098,151 B; Terminal tag `v0.2.3`,
commit `5c8dd322528cc84f68f461ad54a506b9da8fa8d4`, build `0.1.48`; `strata-engine-0.1.48-linux-x86_64-hip-rx7900`,
242,858,823 B, sha256 `13bcf38c…`, `build_id 6f8800aefe27+rocm`, manifest `59a2ac74…`, 581 files; hipBLASLt table
unchanged) and raised `conversation-cache-slots` from `4` to `8`. First start re-ran setup once (~40 s warm, pack/MTP
reused). 0.2.3 clamps image tokens when `vision.max_tokens` is absent (CPU 300, GPU 1024; #3690). (`0.2.3.1` / Control
`0.1.48.1` was a Windows-only hotfix with no Docker image.)

# 20261024 Chat template override and upgrade to 0.2.4.2 (own image with strata-dsv41)

- **Repo chat template override (entrypoint fork change 3)**: Strata extracts `<pack>/tokenizer/chat_template.jinja` at
  pack time and has no `--chat-template-file` flag. Fork change 3 compares `CHAT_TEMPLATE_PATH` (mounted from repo
  `templates/` at `/templates:ro`) with every pack's `tokenizer/chat_template.jinja` before `strata launch` and
  overwrites when different. Verified with `templates/qwen3.8/Flash-Next-chat-template.jinja` (8,952 B).
- **Upgrade to 0.2.4.2 (`strata-image.yml` restored)**: The `0.2.4.x` line (Control `0.1.49.2`) added
  DeepSeek-V4.1-Flash on `strata-dsv41`, published no Docker image after 0.2.3, and shipped `strata-dsv41` only for
  `cuda13` and `hip-rx9070`. Restored `.github/workflows/strata-image.yml` to build Terminal `v0.2.4.2` around the
  signed-feed `0.1.48` Qwen engine package (byte-identical to 0.2.3, `BUILD.json` `7108976a…`) and compile
  `strata-dsv41` for gfx110x via `deploy/strata/build-dsv41.sh`.
- **Pins (0.2.4.2)**: `ghcr.io/hurui200320/llm-ops-strata:0.2.4.2-rocm-gfx110x`, digest
  `sha256:c67827ee283e5e90be31e6c01925727256a26e76aa5b6ee15f57762fa29cf3bc` (1,889,591,459 B, CI run `40100350366`);
  tag `v0.2.4.2`, commit `dfa3f4025c9f1bc4af8cee201b20a2d223061813`; `strata-dsv41 0.1.49.2 build dfa3f4025` (sha256
  `53af2567…`). Because the `0.1.48` engine folder was unchanged, both Qwen lanes kept their 0.2.3 install records and
  served immediately.

# 20261025 DeepSeek-V4.1-Flash lane (dsv41) first deployment and upgrade to 0.2.5

## DeepSeek-V4.1-Flash (`deepseek-v4.1-flash-2bpw`) bring-up on 0.2.4.2

Deployed the IST-DASLab **2.00 bpw** (`IQ2_XXS`) set of DeepSeek-V4.1-Flash (1017B MoE, 40 layers × 384 experts, top-6)
on `strata-dsv41` (speed curve and tuning: [`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md)):

| File | Size (B) | sha256 (matches upstream `setup.py` pin) |
| --- | ---: | --- |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-0000{1..4}-of-00004.gguf` | 39,075,714,624 / 39,388,449,504 / 39,246,891,616 / 20,527,837,312 | `d46efb67…` / `cec1b2c3…` / `9d9c2140…` / `9b790a22…` |
| `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (NVMe-read n-gram table) | 110,595,290,816 | `eb19f30d…` |
| `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` (DSpark drafter) | 7,976,914,240 | `6ba6ac1b…` |
| `mmproj-BF16.gguf` (vision tower; packed from 0.2.7) | 972,208,288 | `9a1ef7c4…` |
| `DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf` (`--engram-meta` only, fetched into `Strata-data/`) | 5,780,589 | `cc15d470…` |

- **Engine discovery in Docker (entrypoint fork change 4)**: With `STRATA_CONTROL_INSTALL=1` /
  `STRATA_INSTALL_DIR=/data/install`, `dsv41_exe()` checks only `/data/install/engine-dsv41/` and
  `/opt/strata/terminal/engine/`. Fork change 4 exports `STRATA_DSV41_EXE` from
  `/opt/strata/components/engine/*/strata-dsv41` when unset.
- **Launch settings & placement**: `CONTEXT=256k` (`--max-context 262144`), no `--kv` flag (`kv=fp16` exits 2), single
  slot. `--ssd-cache` defaults to `avail - 12 GB` (cap 48 GB); after tuning, the lane settled on
  `STRATA_SET="ssd-cache=36 reserve-mib=896"` (never pass `--ssd-cache 0`). `--tp 2` demotes to an auto layer split
  (IQ2_XXS 256-value blocks cannot split `n_ff 2304`): CUDA0 `[0,20]` (2,256/8,064 experts in VRAM) + CUDA1
  `[21,39]`+head (2,097/7,296) = 4,353 experts in VRAM (37.9 GB), 11,007 on CPU (94.9 GB, 16 threads), DSpark
  auto-unloaded. Cold start 48.9 s, warm 43.4 s, stop 1.8 s. Note: llama-swap expands `${PORT}` in **sorted model-ID
  order** (`deepseek-v4.1-flash-2bpw` got `10001`, moving Qwen lanes to `10008`/`10009`).

## Upgrade to 0.2.5 (engine 0.1.50)

- **Pins (0.2.5)**: `ghcr.io/hurui200320/llm-ops-strata:0.2.5-rocm-gfx110x`, digest
  `sha256:0bf409c527a5dd9f008d9392ada2457660b3ec2162acf68df6f4822e9b4a092d` (1,900,632,969 B, CI run `40188533610`);
  tag `v0.2.5`, commit `fa663fabc5c2c568d6035b3ae853b38e1762865a` (Control `0.1.50`);
  `strata-engine-0.1.50-linux-x86_64-hip-rx7900` (243,072,991 B, sha256 `84698dbf…`, manifest `43746aaf…`, `BUILD.json`
  `4b70ff83…`, `build_id 6270e2d9d6bf+rocm`, 580 files); `strata-dsv41 0.1.50 build fa663fabc` (sha256 `3b6aa77f…`);
  hipBLASLt table re-tuned in this package (`sha256 235a26c3…`, 873 B, version `100200`, measured neutral in
  [`notes/qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md)).
- **Changes**: Engine folder move (`0.1.48` → `0.1.50`) triggered a one-time setup pass with no repack/download. Kept
  Qwen lanes installed with `int8` + `kv=fp16` in `STRATA_SET` to avoid rewriting install records. `strata-dsv41` 0.1.50
  added `--experts` (default 6), deterministic chunked prefill from 96 tokens with background warm-up, and opt-in gfx11
  WMMA prompt kernels (`-e STRATA_DS4_HIPK=1`, enabled on the DeepSeek lane for +18.7% pp).

# 20261026 Upgrade to 0.2.6 (strata-dsv41 0.1.51)

Upgraded to `ghcr.io/hurui200320/llm-ops-strata:0.2.6-rocm-gfx110x` (digest
`sha256:46f1ac8203bd5c20bda83a817a10215545cd26d352cccd89dedd9da09315e243`, 1,900,649,052 B, CI run `40313031068`; tag
`v0.2.6`, commit `4b9e05a2a806a3fe24f7b2560d98626c37dade35`, Control `0.1.51`; `strata-dsv41 0.1.51 build 4b9e05a2a`,
sha256 `98243abd…`). Upstream re-shipped the `0.1.50` rx7900 Qwen engine package byte-identical (`84698dbf…`), so all
three lanes kept their install records with no setup pass. `strata-dsv41` 0.1.51 added the `--mmap-experts auto`
copy-all rule (inert here: ~95 GB CPU experts vs 62 GiB RAM, and explicit `ssd-cache=36` disables auto).

# 20261027 Qwen3.8 Flash-Next vision token budget 1024–8192 and VRAM reserve tweak

Setup writes the Qwen picture reader's token budget into `/data/install/strata-iq3_s.json` under `"vision"` (default
1024 on GPU / 300 on CPU; #3690) with no launch flag override. Added **entrypoint fork change 5**: when `VISION_TOKENS`
/ `VISION_MIN_TOKENS` are set, patch `max_tokens` and `min_tokens` in every run config's `vision` section before server
start (forwarded to `strata-vision-hip --max-tokens` / `--min-tokens`). Both Qwen lanes set `VISION_TOKENS=8192` and
`VISION_MIN_TOKENS=1024` (1 token per 1024 px at patch 16 × merge 2, 32-px aligned: 640×480 scales up to 1024, 1080p ≈
2025, 4K 3840×2160 = 8160 tokens; reader warm-up at 2048×2048 moved from 1024 tok / 0.4 s to 4096 tok / 3.5 s) and
raised `vram-reserve-later-mib` from `768` to `1024`.

# 20261028 Upgrade to 0.2.7 (engine 0.1.52, DeepSeek vision & 512k context)

- **Pins (0.2.7)**: `ghcr.io/hurui200320/llm-ops-strata:0.2.7-rocm-gfx110x`, digest
  `sha256:b84c7a5949f6394cb83b5074e0422010cfd1ae82983a9a2ba91ffb9d2fbbf0a1` (1,939,482,478 B on host / 1,505,264,720 B
  in CI, run `40543493857`); tag `v0.2.7`, commit `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d` (Control `0.1.52`);
  `strata-engine-0.1.52-linux-x86_64-hip-rx7900` (245,001,814 B, asset sha256 `2ade169f…`, manifest `a7d13988…`,
  `BUILD.json` `2f9f4176…`, source `f-amd-vision a42cc3d7df70`, 580 files); `strata-dsv41 0.1.52 build 5e8a4cca7`
  (sha256 `9c65ec3a43b7f3d1132bc696e7aa18bb6322ce34925f2cd865403c92a8e8b6ea`); hipBLASLt table reverted to the 0.1.48
  bytes (`1bb8cd2e…`, 845 B, version `100200`, TheRock `7.10.0a20251120`).
- **Build & Host-header guard (`STRATA_ALLOWED_HOSTS=fedora-tuf`)**: Upstream merged PR #3974 (nested `tools/` Docker
  context fix) and added `legacy_models_guard` (#3421) to `entrypoint.sh`. 0.2.7 also added DNS-rebinding `Host` checks
  on keyless servers, returning `403` when llama-swap forwarded `Host: fedora-tuf:8080`; fixed by adding `-e
  STRATA_ALLOWED_HOSTS=fedora-tuf` to `docker-server-strata-rocm`.
- **DeepSeek in-engine vision & `DS4_VISION_TOKENS=1140` (entrypoint fork change 6)**: `strata-dsv41` 0.1.52 packs tower
  metadata from `mmproj-BF16.gguf` (`9a1ef7c4…`) in place (required one setup pass with `-e VISION=on` to replace the
  0.2.6 `"vision": "off"` preset; stop any loaded Qwen lane first so the init smoke does not fail `cudaMalloc`). Images
  are resized to 42-px multiples (patch 14 × merge 3 = 1,764 px/token, span `n_h*(n_w+1)+2`). Fork change 6 writes
  `DS4_VISION_TOKENS=1140` (the tower's 10,240-patch softmax hard cap; +17 MB over the 1024 default) into `ds4.txt` at
  every start:

  | Aspect | 1024 (setup default) | 1140 (this lane) |
  | --- | --- | --- |
  | Square / 16:9 / 1080p / 4K UHD | 1294×1294 (994) / 1756×966 (991) / 1722×966 (968) / 1680×966 (945) | 1378×1378 (1124) / 1840×1008 (1082) / 1806×1008 (1058) / 1764×1008 (1034) |

- **Rollout & DeepSeek 512K context**: All lanes ran one setup pass (`0.1.50` → `0.1.52`); Qwen lanes logged staged HC
  read on CUDA0/1; DeepSeek warm-started in 40.2 s with `--vision on` (4,050 experts in VRAM vs 4,353 on 0.2.6) and
  promoted `CONTEXT=512k` (`--max-context 524288`, `capabilities.context: 524288`, caches `0.24+0.10` → `0.48+0.19 GB`,
  3,994 experts in VRAM, speed-neutral vs 256k; [`notes/deepseek/v41f1017.md`](deepseek/v41f1017.md)).

# 20261030 Upgrade to 0.2.8 (engine 0.1.54), ISTA six-shard GGUF layout, and preset extra mechanics

## Upgrade to 0.2.8 (engine 0.1.54)

- **Pins (0.2.8)**: `ghcr.io/hurui200320/llm-ops-strata:0.2.8-rocm-gfx110x`, digest
  `sha256:476ded1c680e26e888fc84e93eeec97adcf6997be8124655da9340c1d5c1ba29` (1,969,765,328 B, CI run `41015223494`);
  tag `v0.2.8`, commit `870aedd9e1ebe1c91f94da93b59af17729f6c5c0` (Control `0.1.54`);
  `strata-engine-0.1.54-linux-x86_64-hip-rx7900` (250,580,785 B, asset sha256 `4d2833cd…`, manifest `e70b103f…`,
  `build_id 8c1aea4bca1b+rocm`); `strata-dsv41 0.1.54 build 870aedd9e` (sha256 `abc048e9…`); hipBLASLt table
  (`1bb8cd2e…`) and upstream `/opt/strata/entrypoint.sh` (`10fe9c4f…`) unchanged.
- **Engine & server changes**: 16-row verify/batch windows on HIP (`STRATA_WINDOW_MAX_T=auto` → 16, bit-exact);
  `tool_call_recovery` on by default; DeepSeek single-token `<|im_end|>` stop-id fix (#4576/#4137, fixing the "Nerazim"
  early stop); layer-split chunked-prompt deadlock fix (PR #4211); Qwen late-allocation reserve pricing (#4474) and PLE
  lock vs conversation cache (#4518); 120 s wait for leftover engine memory (#4051). `DS4_FAT_ROWS` is CUDA-only
  (`fat_ready()` false under `STRATA_USE_HIP`).
- **Rollout**: One setup pass per lane (`0.1.52` → `0.1.54`, packs reused, `VISION_TOKENS` re-applied on Qwen). Vanilla
  Qwen placed 8,304 + 7,849 = 16,153 VRAM expert slots, 22.87 GiB pinned RAM, lend 3,580; DeepSeek placed 3,994 VRAM
  experts (2,037 + 1,957). All functional smokes and longctx spots passed
  ([`qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md), [`deepseek/v41f1017.md`](deepseek/v41f1017.md)).

## ISTA's 2026-10-27 six-shard DeepSeek GGUF republish (blocked until 0.2.9)

ISTA-DASLab republished its DeepSeek sets on 2026-10-27 in llama.cpp's b11870+ `deepseek41` layout: 5 backbone shards +
shared Engram as shard 6 (`…-00006-of-00006.gguf`), `ple.*` / `engram.token_map` metadata instead of integer tensors,
single `per_layer_token_embd` table, `engram_wkv` → `engram_kv`, `engram_q`/`engram_k` → `engram_gate`, added
`exp_probs_b_vl`, 2-D `attn_output_a`, and `clip.projector_type` `deepseek4v` → `deepseek41v`. While `pack_gguf.py`
already knew those tensor names, `0.2.7`–`0.2.8`'s `setup.py` still pinned `HF_REVISIONS` at `cb02422c…` (2026-10-24)
and `DSV41_ISTA_SHARD` as `-of-00004.gguf`, so converting or replacing the shards in place on 0.2.8 broke the
filename/offset-bound `tensors.txt` pack and failed recovery setup until 0.2.9.

## A/B mechanics: preset `extra` accumulates `--set` keys

`strata launch --set KEY=VALUE` (driven from `STRATA_SET`) merges keys into
`<data>/config/strata/launch-presets/docker.json` → `"extra"`, and every start re-derives engine argv/env from it.
Removing a key from `STRATA_SET` in `deploy/llama-swap.config.yaml` (or clearing derived `state/strata/launch/docker.json`)
does **not** revert it—the next launch re-adds it from `preset["extra"]` (superseded in 0.2.11 by `arg_layers.py`).
Always verify `/proc/<pid>/environ` and the engine log's `engine started:` line during A/B runs, and pre-warm with a
retry loop because the first request starting a stopped lane through llama-swap can race the server and return `502`.

# 20261031 Upgrade to 0.2.9 (engine 0.1.55) — ISTA's six-shard DeepSeek layout adopted

Upgraded to `0.2.9`/`0.1.55` and swapped the DeepSeek model folder to ISTA's six-shard 2.00 bpw set (described in the
20261030 entry above).

- **Pins (0.2.9)**: `ghcr.io/hurui200320/llm-ops-strata:0.2.9-rocm-gfx110x`, digest
  `sha256:4d369491ccd53fa878cc0b871b52789820981a6059c51c533cce956476fb50a2` (1,971,106,262 B on host / 1,530,632,544 B
  in CI, run `41200983707`); tag `v0.2.9`, commit `c62d5e3a15b40ea51c96aae9e8d36928965f53ab` (Control `0.1.55`);
  `strata-engine-0.1.55-linux-x86_64-hip-rx7900` (250,890,402 B, asset sha256 `5cfedda6…`, manifest `b3f45acb…`,
  `build_id 99866a47a2b8+rocm`, 580 files); `strata-dsv41 0.1.55 build c62d5e3a1` (sha256 `7738aa7f…`); hipBLASLt table
  and entrypoint unchanged.
- **Engine changes**: Qwen 0.1.55 is functionally neutral on gfx1100 (NVIDIA/RDNA4 changes only). `strata-dsv41` 0.1.55
  adds `DSV41_ISTA6_FILES` / `dsv41_layout_of()` / `dsv41_pack_layout()` (#4348, #4632) and accepts the in-place
  converted Engram (`3dac4c54…` via `DSV41_SHA_ALSO`).
- **Six-shard GGUF swap (`/mnt/unenc-xfs/models/strata/ISTA-DASLab-dsv41f1017/`, `-of-00004` files removed)**:

  | File | Size (B) | sha256 (`DSV41_ISTA6_FILES`) |
  | --- | ---: | --- |
  | `…-IQ2_XXS-0000{1..5}-of-00006.gguf` | 5,780,704 / 39,535,781,184 / 39,736,712,000 / 39,036,908,384 / 19,923,784,832 | `f356f198…` / `25af3785…` / `8168d099…` / `9eb9b4de…` / `7e35f6b4…` |
  | `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (converted) | 110,595,290,816 | `3dac4c54…` |
  | `dspark-…-Q8_0.gguf` / `mmproj-BF16.gguf` (kept) | 7,976,914,240 / 972,208,288 | `6ba6ac1b…` / `9a1ef7c4…` |

- **Rollout**: Rebuilt the DeepSeek pack (`engram.bin` 517,980 B, `vision.max_image_tokens 1140`), reproduced the 0.2.8
  512k placement (3,504 store slots / 36.0 GB, 2,037 + 1,957 = 3,994 experts in VRAM, 11,366 on CPU), and passed all
  smokes.

# 20261103 Upgrade to 0.2.10 (engine 0.1.56) — DFlash2 drafter on the Qwen lanes

- **Pins (0.2.10)**: `ghcr.io/hurui200320/llm-ops-strata:0.2.10-rocm-gfx110x`, digest
  `sha256:453e77dbc0dae27e0f057267b74ce48a4abc424d403596cd0747394d22283f7a` (2,317,806,899 B, CI run `41591858905`);
  tag `v0.2.10`, commit `ffa80d73133ea5c548aefe57889a16e8085a938f` (Control `0.1.56`);
  `strata-engine-0.1.56-linux-x86_64-hip-rx7900` (300,041,962 B, asset sha256 `b57768c8…`, manifest `094d78df…`,
  `build_id c75ab7655503+rocm`, 583 files including inert `strata-split-2/3/4-amd`); `strata-dsv41 0.1.56 build
  ffa80d731` (sha256 `8febdca0a968ef4079ef702ad477d1a164e31a1af3fa3b2a99c186b26cd3a013`); upstream `entrypoint.sh`
  sha256 `0c660798…` (+`RETUNE` #4779); fork adds **change 7** (`DRAFTING` → `strata launch --drafting`).
- **Engine changes**:
  - Split engine (`strata-split-*-amd`) is inert here (Auto requires 4× gfx1201, non-IQ3_S, ≤128K ctx).
  - Qwen 0.1.56 adds DFlash2 drafter (`--draft dflash2`) and serve fixes #4677 (parked continuation 4/16 → 0/16 failed
    on Linux HIP), #4590 (reply-only penalties), #4701/#4751/#4663/#4700 (tool-call parsing), and #4846 (PLE lock
    release under RAM pressure).
  - `strata-dsv41` 0.1.56 keeps the vision tower in pinned host RAM between pictures on AMD dGPUs (~320 MiB taken from
    `--reserve-mib` only during picture turns; lane raised `reserve-mib=896` → `900` to match
    `DSV41_VISION_RESERVE_MIB`), lifting VRAM experts from 3,994 to **4,041** (2,082 + 1,959;
    [`deepseek/v41f1017.md`](deepseek/v41f1017.md)). Also adds `dq_rows_kernel2` default on gfx11 and #4772.
- **DFlash2 A/B & same-day revert to MTP**: 0.2.10 makes DFlash2 Auto on gfx1100, but `--pipeline-windows 2` blocks it
  at serve time while `needs_pack()` still fetches the 1.2 GB drafter (`Strata-data/dflash2/`, ~860 MB `q4_0-selq4`
  pack, 715 MiB VRAM vs MTP's 1,082 MiB, +6.4% VRAM expert slots to 16,826). Tested DFlash2 explicitly (`DRAFTING=dflash2`
  without `pipeline-windows=2 spec-min-p=0.7`): decode was **16–18% slower** than MTP on 0.1.56 (54.80/52.65 vs
  65.00/64.30 tg at 60/85%, acceptance 0.44/0.42 vs 0.65; [`qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md)). Reverted
  both Qwen lanes to `DRAFTING=mtp` with `pipeline-windows=2 spec-min-p=0.7` (reproducing 7,959 + 7,849 slots, 23.5 GiB
  RAM copy, lend 3,581).

# 20261104 DeepSeek retires from strata (workflow build removed, fork stripped to five changes)

With `maya-ds4` serving the ISTA set faster on every axis ([notes/maya.md](maya.md) 20261104;
[notes/deepseek/v41f1017.md](deepseek/v41f1017.md) 20261104), `strata-dsv41` has no consumer left:
- Removed `include_dsv41` and all CMake/TheRock/llama.cpp/`build-dsv41.sh` compilation steps from
  `.github/workflows/strata-image.yml`, and deleted `deploy/strata/build-dsv41.sh`. The workflow now assembles
  upstream's signed-feed package only (keeping the pinned fetch, gfx1100 hipBLASLt table gate, smoke, and GHCR push).
- Stripped `deploy/strata/entrypoint-local.sh` from seven changes to **five** (removed `STRATA_DSV41_EXE` and
  `DS4_VISION_TOKENS`, renumbering `--no-api-key`, false-warning suppression, chat-template override, `VISION_TOKENS`,
  and `DRAFTING` as 1–5).
- Deleted `/mnt/unenc-xfs/strata/ISTA-DASLab-dsv41f1017` on the runner, moved the shards to
  `/mnt/unenc-xfs/models/maya/ISTA-DASLab-dsv41f1017/`, and removed the unused dspark GGUF and extra mmproj (~10.4 GiB
  freed).

# 20261106 Upgrade to 0.2.11 (engine 0.1.57) — serve/tool-parser round

- **Pins (0.2.11)**: `ghcr.io/hurui200320/llm-ops-strata:0.2.11-rocm-gfx110x`, digest
  `sha256:5afbc76626af68c01fa1c6518149f61b576a56288feba5b1f9c9d3e474f5533d` (2,350,639,469 B, CI run `42176698284`);
  tag `v0.2.11`, commit `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d` (Control `0.1.57`);
  `strata-engine-0.1.57-linux-x86_64-hip-rx7900` (314 MiB tarball, 583 files, manifest `b82bd673…`, `build_id
  0bdcef4682c5+rocm`, `dsv41: false`); hipBLASLt table (`1bb8cd2e…`) and upstream `entrypoint.sh` (`0c660798…`, 5 fork
  changes) unchanged. Added `deploy/strata/patches/0.2.11-metal-tools-context.patch` (upstream `5d37280d`) because
  v0.2.11's `package_terminal.py` copies `metal/tools/*.py` before `docker/build.sh` staged them (drop patch on next
  release).
- **What changes for this box**:
  - **Serve & tool parser (0.1.57)**: #3961 (stream ending without end event no longer kills next request), #4271
    (stalled streaming client releases turn after 60 s, `STRATA_STREAM_WRITE_TIMEOUT_S`), #4479 (blocked console), #4738
    (control-line wake-up), #5044 (refused batch admission / non-UTF-8 output), #4792 (failed-start race), #3443
    (conversation-park log lock), #2518 (branching follow-up returns to its chat slot), and #5040 (streamed tool args
    containing closing tags no longer truncate in the parser).
  - **Launch precedence (`ad142b46`, `strata_control/arg_layers.py`)**: Explicit launch values now win over persisted
    overrides, calibrated values, and Auto; `STRATA_SET` keys land in `user_args` + `launch_set` without sticky
    `extra`-accumulation.
  - **DFlash2 & split engine**: DFlash2 still refuses beside `--pipeline-windows` (MTP stays); Auto split engine still
    requires 4× gfx1201 and ≤128K ctx.
- **Rollout & verification (2026-11-06)**: Both Qwen lanes ran one setup pass (`0.1.56` → `0.1.57`, packs reused,
  `max_tokens=8192, min_tokens=1024` re-applied). Placement: vanilla `7,959 + 7,727` slots (14.44 + 15.47 GiB), 23.75
  GiB RAM copy, lend 3,582, 1,818/1,120 MiB free; SC117 `7,905 + 7,691` (15,596 slots), 23.91 GiB RAM copy, lend 3,582
  (MTP draft layer 1,348 MiB). Passed all llama-swap smokes, in-image test suites (`serve.test_serve_robustness` 11/11,
  `serve.test_tool_corpus` 4/4, `OpenAIChunksTerminal` 10/10), a live stalled-reader cancellation test, and the 262144
  longctx spot ([qwen3.8/Flash-Next.md](qwen3.8/Flash-Next.md) 20261106). Note: when a `write_file` tool call's content
  is a multi-line document listing `</parameter>`, `</function>`, and `</tool_call>`, the *model itself* stops before
  the tag block and emits extra `</tool_call></invoke>` closers (not a parser bug).

# 20261108 Upgrade to 0.2.12 (engine 0.1.58) — tool-parser round, pipeline-windows fix, HF template incident

- **Pins (0.2.12)**: `ghcr.io/hurui200320/llm-ops-strata:0.2.12-rocm-gfx110x`, digest
  `sha256:5383b15fb5215ce1c470ec63b9b0a18301cad95112f12b703e7a77ef459fcbf3` (1,848,057,072 B in CI /
  2,351,866,055 B on host, CI run `42387884421`); tag `v0.2.12`, commit
  `146df3c468739147f520ac3816de806f71232571` (Control `0.1.58`);
  `strata-engine-0.1.58-linux-x86_64-hip-rx7900` (329,339,566 B, asset sha256 `7d83a773…`, 583 files, manifest sha256
  `e3b1af1f89847912db333e74701b2380012d8e5390bd56b0cc9f81e5e4b3fe89`, `build_id 7be42963e6be+rocm`). hipBLASLt table
  bytes unchanged (`1bb8cd2e232ce1dec5dc0f303e5bd3b32abd9444330a4428a995ad4db8fb97be`, version `100200`) and upstream
  `entrypoint.sh` unchanged (`0c660798…`, the fork keeps its five changes). v0.2.12 contains the v0.2.11 build-context
  fix `5d37280d`, so `deploy/strata/patches/` and the workflow's patch step were removed.
- **What changes for this box**: the tool-parser round (#5409 `<function=tools NAME>`, #5276 misspelt openers, #5281 an
  MCP tool by its last segment, #4828 forced `tool_choice` with several tools, #4832 Anthropic results by
  `tool_use_id`, #4622 `undeclared_tool_calls`, default `text`) and the #5149/#5147 fix for a first-request hang with
  `--pipeline-windows 2` on two AMD cards (our lane shape; same windows, tokens unchanged). Serve reports now count
  admission time, so short requests read lower t/s without being slower. The Mac/Strix Halo/gfx12/CUDA speed work does
  not apply; upstream measured the closest arm (2× A6000, IQ3_S, layer split) at +0.38%. Upstream did **not** run the
  rx7900 package on hardware this release.
- **`STRATA_ADAPT_FETCH` is inert here.** The engine turns it off when `--pipeline-windows`, a layer split, or the
  resident RAM complement is active; a vanilla start with `-e STRATA_ADAPT_FETCH=2` logged
  `strata serve: STRATA_ADAPT_FETCH is off (a layer split)`, so no A/B is possible on these lanes.
- **Template incident (fixed in the same change)**: the 2026-11-07 template refresh (`6eb3392`,
  `templates/fetch-templates.sh`) brought Hugging Face's current Qwen3.8-Flash-Next template, which wraps each
  assistant turn in `{% generation %}` / `{% endgeneration %}` (training-time assistant-token masks; they render
  nothing). Strata compiles the template with plain Jinja and has no such extension: the engine loaded, then
  `serve.server` exited 1 at `ChatTemplate(...)` with `TemplateSyntaxError: Encountered unknown tag 'generation'`, and
  `strata launch` reported "The engine stopped while it was loading the model". The whole-line markers are removed into
  `templates/qwen3.8/Flash-Next-chat-template-fixed.jinja` (generated by `fetch-templates.sh`, whose header says so);
  both lanes' `CHAT_TEMPLATE_PATH` point at that copy, and the fetched file stays upstream's. The rest of the update is
  a refactor (the `<|im_start|>role\n` prefix hoisted out of the branches, `<|im_end|>\n` split around the marker);
  old vs fixed rendered identically on 22 cases (plain, thinking on/off, reasoning, tools, multi-step tools,
  image/video, generation flags, error paths).
- **Rollout & verification (2026-11-08)**: both lanes ran one setup pass (`0.1.57` → `0.1.58`, packs reused,
  `VISION_TOKENS` re-applied, `Chat template: replaced …-fixed.jinja` logged). Placement, vanilla: 7960 + 7727 slots
  (14.44 + 15.47 GiB), 23.75 GiB RAM copy, lend 3582, 1818/1120 MiB free — the 0.2.11 figures; SC117: 8252 + 7691
  slots (14.97 + 15.40 GiB), 23.28 GiB RAM copy, lend 3584, 1814/1120 MiB free. Startup logged
  `prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)` and `--pipeline-windows 2: two verifiers
  per stage`. Smokes: chat with reasoning split, declared tool call, multi-turn tool result, vision ("Red" on a 1×1
  red PNG), `chat_template_kwargs.preserve_thinking: false`, `/props` caps all true, in-image suites
  (`serve.test_serve_robustness` + `serve.test_tool_corpus` 14/14, `OpenAIChunksTerminal` 10/10, the 0.2.12
  tool/accounting tests 61/61), graceful stop 2.9 s. Longctx spot (speed record:
  [qwen3.8/Flash-Next.md](qwen3.8/Flash-Next.md) 20261108): speed-neutral against the 0.2.11 control.
- **Cutover (2026-11-08, after the template-fix commit was pulled on fedora-tuf)**: llama-swap starts both lanes on
  the new image with `CHAT_TEMPLATE_PATH` at the `-fixed.jinja` copy — vanilla answered chat, a declared tool call,
  the `:npt` alias and vision ("Red" on the 1×1 PNG); SC117 answered a tool call and vision; both containers
  `(healthy)` on `0.2.12-rocm-gfx110x`. The pack templates are already the fixed copy, so the lane start logs no
  further swap.

# 20261110 Upgrade to 0.2.13 (engine 0.1.59) — parser from the template, engine-side hipBLASLt, #5327 file-cache credit

- **Pins (0.2.13)**: `ghcr.io/hurui200320/llm-ops-strata:0.2.13-rocm-gfx110x`, digest
  `sha256:a93ef7902bc52c68ab164e3d3581121dc3f18552befc55c7d3d881196e7ae3e0` (2,348,902,878 B on host, CI run
  `42657519003`); tag `v0.2.13`, commit `d2f096475817d4666c364c820b0ca830f751cfc5` (Control `0.1.59`);
  `strata-engine-0.1.59-linux-x86_64-hip-rx7900` (asset sha256
  `9ba0e3e33a1dad14870df46d9fc70e7cd2e5d5690196c8e4b39cc3aa1b261059`, 583 files, manifest sha256
  `df4a624b9493086e57e8195fb59a994e26c3b38226c456a7d4d0859a79493367`, `BUILD.json` source `f-0213-rc
  712fec6b4297`, `build_id 712fec6b4297+rocm`, variant `rx7900`, `dsv41: false`). hipBLASLt table bytes unchanged
  (`1bb8cd2e232ce1dec5dc0f303e5bd3b32abd9444330a4428a995ad4db8fb97be`, 845 B, version `100200`) and upstream
  `docker/` (entrypoint.sh `0c660798…`, build.sh, Dockerfile, fetch-engines.py) unchanged from v0.2.12, so the
  five fork changes and the workflow's build steps carried over untouched.
- **What changes for this box**:
  - **Tool parser is picked from the chat template (`serve/toolformat.py`, `parser_for_template`)**: the XML
    form every Strata model ships — our `Flash-Next-chat-template-fixed.jinja` included — keeps the same
    `OutputParser`; only JSON-form templates (Qwen3/Hermes) get the new `JsonOutputParser`, and DSML stays
    family-selected. No behavior change on either lane (declared/multi-turn tool smokes pass).
  - **hipBLASLt table discovery moved into the engine (`src/prefill/gemm.cu`, 343b8eae)**: the server no longer
    writes `STRATA_HIPBLASLT_TUNING` into the run config; the engine takes
    `tuning/<arch>-hipblaslt-<version>.txt` beside itself on gfx1151 by itself and on another card with
    `STRATA_HIPBLASLT_TABLE=on` (the container env both lanes already pass, inherited by the engine). Verified in
    the engine log on both lanes: `prefill gemm: hipBLASLt tuning enabled (26 rows, gfx1100, version 100200)`.
    The server's own `hipBLASLt tuning table: …` line is no longer visible in either the container log or
    `docker.engine.log`.
  - **Engine-owned defaults (d1121b89, f-clean-1c)**: the parallel stack, GDN stack, Q8_0 hyper-connection read,
    Metal fp16 windows and the gfx1151 table path are no longer injected by the server; none of them ever
    applied to HIP gfx1100 (Linux-CUDA / gfx12 / Metal only), so the lanes are unaffected.
  - **#5327 / PR #5531 (28bd0185…)**: on Linux an expert-arena shortfall the free pages + file cache can cover
    loads again (the #4051/#4846 refusal stays for anonymous-RAM shortfalls and cgroup caps;
    `STRATA_ARENA_FILE_CACHE=0` opts out). The start log shows the credit in action: `the file tier reads through
    the file cache (55.4 GiB available, 28.0 GiB of it still to be taken by the RAM copy, …)`.
  - **fused_gr dynamic-shared-memory opt-in set per device, not per process** (269f3385 + b4e9ac7c): the 2-card
    layer split runs the single-token down kernel on both cards; Flash-Next's hidden size makes the dynamic-smem
    path inert, so no placement change.
  - DFlash2 per-device defaults (window 8 / conf 0.55) are Strix Halo and Blackwell ≥48 GB only; gfx1100 keeps
    the 0.2.12 values and both lanes stay on MTP (`--pipeline-windows 2` still blocks DFlash2).
  - Qwen4 groundwork (signed-feed model list, `strata-q4-<shape>` components, `strata --geometry`): inert — the
    0.2.13 feed carries neither a model list nor a shape engine; `strata --geometry` prints `flash_next` for the
    rx7900 engine. Upstream did **not** run the Linux rx7900 package on hardware this release.
- **Rollout (2026-11-10)**: pulled on fedora-tuf (`6f5e30f` → `4945b57`); llama-swap watchconfig reloaded the
  config at 00:53:12. Both lanes ran one setup pass (`0.1.58` → `0.1.59`): the install record was remade because
  its `vision.exe` named the 0.1.58 folder, packs and MTP were reused (no download), and the first start warned
  that `VISION_TOKENS` is applied on the next start — the reload for the longctx run then logged
  `Vision budget: strata-iq3_s.json: max_tokens=8192, min_tokens=1024`, and the reader warmed at 4096 tokens.
  Vanilla first request 200 in 44 s (setup+load), SC117 in 49 s; a later stop/start reload took 45 s.
- **Placement (with the 8192 vision budget applied)**: vanilla `7,959 + 7,727` slots (14.44 + 15.47 GiB), 23.75
  GiB RAM copy, lend 3,582, 1,818/1,120 MiB free — the 0.2.12 figures exactly; SC117 `8,253 + 7,691` (14.97 +
  15.40 GiB), 23.3 GiB RAM copy, lend 3,584, 1,812/1,120 MiB free (0.2.12: 8,252 + 7,691 / 23.28 / 3,584). (The
  setup start, still on setup's default 1024-token reader, placed 8,304 slots on CUDA0; the serving start with
  the lane's own config returns to 7,959.)
- **Verification**: both lanes 6/6 llama-swap smokes (chat with the reasoning split, declared tool call,
  multi-turn tool result, vision "Red" on a 1×1 PNG, `:npt`, `/props` n_ctx 262144 + all caps true); pack
  templates equal the repo `-fixed.jinja` (sha256 `94c3110e…`, no swap logged); in-image suites
  (`serve.test_serve_robustness` + `serve.test_tool_corpus` + `OpenAIChunksTerminal` 24/24, tool/accounting
  family 124/124, `serve.test_tool_format` 12/12 with the source fixtures mounted); graceful stop 2.7 s (0.2.12:
  2.9 s). **Image gap (upstream)**: `serve/test_tool_format.py` reads `tools/qwen4_intake_fixtures/*.json.gz`,
  which `packaging/package_terminal.py` does not copy into the image, so 7 of its tests error in-image without
  the fixtures; runtime serving does not read them.
- **Longctx spot (speed record: [qwen3.8/Flash-Next.md](qwen3.8/Flash-Next.md) 20261110)**: 60% pp 2,100.9 / tg
  66.85, 85% pp 2,185.9 / tg 64.90 (accept 0.68) against the 0.2.12 control 2,080.2/66.40 and 2,161.3/64.20 —
  speed-neutral, within +1%. `deploy/strata/README.md` was updated in place (pins, engine-side table discovery,
  engine-log check line).
