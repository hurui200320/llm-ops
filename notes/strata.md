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
  `sha256:5383b15fb5215ce1c470ec63b9b0a18301cad95112f12b703e7a77ef459fcbf3` (1,848,057,072 B in CI / 2,351,866,055 B
  on host, CI run `42387884421`); tag `v0.2.12`, commit `146df3c468739147f520ac3816de806f71232571` (Control `0.1.58`);
  `strata-engine-0.1.58-linux-x86_64-hip-rx7900` (329,339,566 B, asset sha256 `7d83a773…`, manifest
  `e3b1af1f89847912db333e74701b2380012d8e5390bd56b0cc9f81e5e4b3fe89`, `build_id 7be42963e6be+rocm`, 583 files);
  hipBLASLt table (`1bb8cd2e…`, `100200`) and upstream `entrypoint.sh` (`0c660798…`, 5 fork changes) unchanged. Dropped
  `deploy/strata/patches/0.2.11-metal-tools-context.patch` (upstream `5d37280d` is included in v0.2.12).
- **Engine & server changes**: Tool-parser fixes (#5409 `<function=tools NAME>`, #5276 misspelt openers, #5281 MCP tool
  by last segment, #4828 forced `tool_choice` with several tools, #4832 Anthropic results by `tool_use_id`, #4622
  `undeclared_tool_calls` default `text`) and #5149/#5147 fix for a first-request hang with `--pipeline-windows 2` on
  two AMD cards. Serve reports now count admission time. `STRATA_ADAPT_FETCH` is inert here (disabled whenever
  `--pipeline-windows`, a layer split, or the resident RAM complement is active).
- **Template incident (`{% generation %}` tags)**: The 2026-11-07 template refresh (`6eb3392`,
  `templates/fetch-templates.sh`) pulled Hugging Face's Qwen3.8-Flash-Next template containing `{% generation %}` /
  `{% endgeneration %}` training-time masks, causing `serve.server` to exit 1 at `ChatTemplate(...)`
  (`TemplateSyntaxError: Encountered unknown tag 'generation'`). Fixed in `fetch-templates.sh` by stripping those
  whole-line markers into `templates/qwen3.8/Flash-Next-chat-template-fixed.jinja` (verified identical to the old
  template across 22 cases) and pointing both lanes' `CHAT_TEMPLATE_PATH` at `-fixed.jinja`.
- **Rollout & verification (2026-11-08)**: One setup pass per lane (`0.1.57` → `0.1.58`, packs reused, `VISION_TOKENS`
  and `-fixed.jinja` applied). Placement: vanilla `7,960 + 7,727` slots (14.44 + 15.47 GiB), 23.75 GiB RAM copy, lend
  3,582, 1,818/1,120 MiB free; SC117 `8,252 + 7,691` slots (14.97 + 15.40 GiB), 23.28 GiB RAM copy, lend 3,584,
  1,814/1,120 MiB free. Passed all llama-swap smokes (chat/reasoning, tool call, multi-turn tool, vision, `:npt`,
  `/props`), in-image test suites (14/14, 10/10, 61/61), graceful stop (2.9 s), and the 262144 longctx spot
  (speed-neutral vs 0.2.11; [qwen3.8/Flash-Next.md](qwen3.8/Flash-Next.md) 20261108).

# 20261110 Upgrade to 0.2.13 (engine 0.1.59) — parser from template, engine-side hipBLASLt, #5327 file-cache credit

- **Pins (0.2.13)**: `ghcr.io/hurui200320/llm-ops-strata:0.2.13-rocm-gfx110x`, digest
  `sha256:a93ef7902bc52c68ab164e3d3581121dc3f18552befc55c7d3d881196e7ae3e0` (2,348,902,878 B on host, CI run
  `42657519003`); tag `v0.2.13`, commit `d2f096475817d4666c364c820b0ca830f751cfc5` (Control `0.1.59`);
  `strata-engine-0.1.59-linux-x86_64-hip-rx7900` (asset sha256
  `9ba0e3e33a1dad14870df46d9fc70e7cd2e5d5690196c8e4b39cc3aa1b261059`, manifest
  `df4a624b9493086e57e8195fb59a994e26c3b38226c456a7d4d0859a79493367`, `BUILD.json` source `f-0213-rc 712fec6b4297`,
  `build_id 712fec6b4297+rocm`, 583 files, `dsv41: false`); hipBLASLt table (`1bb8cd2e…`, `100200`) and upstream
  `docker/` (`entrypoint.sh` `0c660798…`, 5 fork changes) unchanged.
- **Engine & server changes**:
  - **Template-selected tool parser (`serve/toolformat.py`, `parser_for_template`)**: XML templates (including our
    `Flash-Next-chat-template-fixed.jinja`) keep `OutputParser`; JSON templates get `JsonOutputParser`.
  - **Engine-side hipBLASLt discovery (`343b8eae`, `src/prefill/gemm.cu`)**: Server no longer writes
    `STRATA_HIPBLASLT_TUNING` into the run config; with container env `STRATA_HIPBLASLT_TABLE=on`, the engine loads
    `tuning/<arch>-hipblaslt-<version>.txt` beside itself directly (logged in `docker.engine.log`).
  - **#5327 / PR #5531 (`28bd0185`)**: On Linux an expert-arena shortfall covered by free pages + file cache loads again
    (`STRATA_ARENA_FILE_CACHE=0` opts out; #4051/#4846 anonymous-RAM/cgroup refusals remain).
  - **Inert here**: Engine-owned defaults (`d1121b89`), per-device `fused_gr` dynamic-smem opt-in (`269f3385`/`b4e9ac7c`),
    Strix Halo/Blackwell DFlash2 defaults, and Qwen4 groundwork (`strata --geometry` prints `flash_next`).
- **Rollout & verification (2026-11-10)**: One setup pass per lane (`0.1.58` → `0.1.59` due to `vision.exe` path;
  packs/MTP reused; reload applied `max_tokens=8192, min_tokens=1024`, warm reload 45 s). Placement: vanilla `7,959 +
  7,727` slots (14.44 + 15.47 GiB), 23.75 GiB RAM copy, lend 3,582, 1,818/1,120 MiB free; SC117 `8,253 + 7,691`
  (14.97 + 15.40 GiB), 23.3 GiB RAM copy, lend 3,584, 1,812/1,120 MiB free. Passed 6/6 llama-swap smokes, in-image
  suites (`test_serve_robustness` + `test_tool_corpus` + `OpenAIChunksTerminal` 24/24, tool/accounting 124/124,
  `test_tool_format` 12/12 with `tools/qwen4_intake_fixtures/*.json.gz` mounted—omitted from the image by
  `package_terminal.py`, unused at runtime), graceful stop (2.7 s), and the 262144 longctx spot (60% 2,100.9/66.85,
  85% 2,185.9/64.90 vs 0.2.12's 2,080.2/66.40 and 2,161.3/64.20; [qwen3.8/Flash-Next.md](qwen3.8/Flash-Next.md)
  20261110).

# 20261116 Upgrades to 0.2.14.1 (engine 0.1.60) and 0.2.15 (Control 0.1.61)

## 0.2.14.1 (engine 0.1.60) — Strata fused prompt kernels, split-trim default, equal prompt cuts

- **Split pin & build fix**: `v0.2.14.1` is a Control-only follow-up (`0.1.60.1`) with no GitHub release object, so the
  workflow pinned `STRATA_REF: v0.2.14.1` with `ENGINE_RELEASE: v0.2.14` (upstream `v0.2.14` commit `09b60388…`).
  Terminal dropped `llama.cpp` in favor of vendored `third_party/strata-gguf-py` (`tools/strata_gguf.py` default), and
  upstream `entrypoint.sh` (`1e6e0290…`) dropped the llama.cpp seed link (dangling `/data/install/third_party/llama.cpp`
  symlinks removed from both lane trees at cutover). Because `docker/build.sh`'s `paths=` list omitted
  `third_party/strata-gguf-py` (first CI run `43801311873` failed with `FileNotFoundError`), our workflow pre-stages
  that folder into `--ctx` before calling `build.sh`.
- **Pins (0.2.14.1)**: `ghcr.io/hurui200320/llm-ops-strata:0.2.14.1-rocm-gfx110x`, digest
  `sha256:791b04211ffbd256a0bd46c666516860ae4bdbc82fe14d75db7a87a68eb6900d` (1,811,989,575 B in CI, run `43802234345`,
  job `133084946753`); tag `v0.2.14.1`, commit `3af562d541ec76c8da63d568248a919c8a54906b` (Control `0.1.60.1`, app
  `0.2.14`); `strata-engine-0.1.60-linux-x86_64-hip-rx7900` (322,924,154 B, 583 files, manifest
  `09cb87e82240c40099ade15e6c8ec34ba23029fb1eeeab8c607bfd2ee787568b`, `BUILD.json` source `f-p4-rc 00157e14b534`);
  hipBLASLt table unchanged (`1bb8cd2e…`, `100200`).
- **Engine & server changes**:
  - **Fused prompt expert kernels**: llama.cpp MMQ kernels are no longer compiled in (`-DSTRATA_MMQ=ON` is source-only).
  - **Layer-split pricing (#4522/#4549)**: Dense-weight trim is now default (`CUDA0 keeps only its own layers' dense
    weights (1.85 GiB in all): CUDA0 0-25 (55 MiB freed)` credited to card 0), while card 1 reserves drafter headroom
    (`+256 MiB for what is allocated after the cache`) and prices later-stage Q8_0 HC rows and PLE keys.
  - **Equal prompt cuts & vision**: Long prompts split into equal pieces (≥5,400 tok for 2 pieces, ≥6,144 for 3+ at 8K+
    chunks). GPU vision holds BF16 matrices as F16 with F16 inputs and bounds context retries (18 s / 12 tries).
  - **Closed TODOs**: `STRATA_PF_P1A` dropped (hard-gated on `dev_info().g1151` in `src/prefill/moe_fused_iq.cu` and
    changes answers); `"draft_costs": true` dropped (Metal-only default, −1.5% CUDA code/chat, no env/flag path).
    `strata-dsv41` sources were removed upstream (already retired here 20261104).
- **Rollout, placement & verification**: One setup pass per lane (`0.1.59` → `0.1.60`, packs/MTP reused, vision 8192
  applied on reload). Vanilla placed **8,253 (14.97 GiB) + 7,609 (15.25 GiB) = 15,862 slots** (+176 net vs 0.2.13's
  7,959 + 7,727: CUDA0 +294 from #4522 trim credit, CUDA1 −118 from #4549 pricing), **22.0 GiB** RAM copy (vs 23.75),
  lend **2,816 slots all RAM-backed** (vs 3,582; no streamed-slot warning), zero GPU-pause warnings. Passed 6/6 smokes,
  in-image suites (80/80 and 28/28), and the 262144 longctx spot ([qwen3.8/Flash-Next.md](qwen3.8/Flash-Next.md)
  20261116).

## 0.2.15 (Terminal 0.2.15 / Control 0.1.61, engine stays 0.1.60) — serve/tool round, two build fixes

- **Pins (0.2.15)**: `ghcr.io/hurui200320/llm-ops-strata:0.2.15-rocm-gfx110x`, digest
  `sha256:9d866b9ec176c5b28a17ddc0c637d4ed7bf8ef3701f009a77509e67cbdf9fe42` (2,307,744,695 B on host, CI run
  `43854371241`, job `133254736174`, commit `2a178e0`); `STRATA_REF` and `ENGINE_RELEASE` both `v0.2.15`, commit
  `c77572806996349b49995e4f98d41b5294d36a21` (Control `0.1.61`, app `0.2.15`); engine package unchanged from 0.2.14.1
  (`strata-engine-0.1.60-linux-x86_64-hip-rx7900`, manifest `09cb87e8…`, referenced via absolute URL into `v0.2.14` in
  feed sequence 127); hipBLASLt table (`1bb8cd2e…`) and `entrypoint.sh` (`1e6e0290…`, 5 fork changes) unchanged.
- **Serve changes & carried build fixes (expected drop in 0.2.16)**:
  - **Serve fixes**: #5979 (`serve/frontend.py` parses declared boolean tool args written `True`/`False` as JSON `bool`,
    live-verified), #6055 (`serve/server.py` returns `503 server_draining` during stop drain; `docker stop -t 120` takes
    2.8 s), and #5601 (`strata_control/runtime.py` treats empty owner state file as not ready yet).
  - **Carried build fixes**: (1) `docker/build.sh` still omits `third_party/strata-gguf-py` from `paths=`, so the
    `--ctx` pre-stage stays; (2) `packaging/binary_deps.py` shrank `LINUX_OS` to glibc + libgcc/libstdc++, failing CI
    run `43844311473` on `librocm_sysdeps_numa.so.1: needs libatomic.so.1` (declared in `SYSTEM-LIBS.txt` and installed
    in the runtime stage via `libatomic1`). Fixed via
    [`deploy/strata/patches/0.2.15-libatomic-verify.patch`](../deploy/strata/patches/0.2.15-libatomic-verify.patch).
- **Rollout & verification**: Because the `0.1.60` engine is byte-identical, both lanes started with **no setup pass**
  (vanilla 51.0 s cold / 50.1 s warm, `30.2 GiB` GPU caches, `22.0 GiB` RAM copy, lend `2,816`; SC117 56.6 s, `30.1 GiB`
  GPU caches, `22.1 GiB` RAM copy, lend `2,817`). Passed all llama-swap smokes (including the #5979 boolean tool check)
  and in-image test suites (**140/140 OK** with `tools/qwen4_intake_fixtures` mounted); longctx skipped (engine
  unchanged from 0.2.14.1).

# 20261119 Upgrade to 0.2.16 (engine 0.1.62) — Strata's own engine code, folder layout move, tool-parser round

- **Pins (0.2.16)**: `ghcr.io/hurui200320/llm-ops-strata:0.2.16-rocm-gfx110x`, digest
  `sha256:6a4df9df58e342e968daa6319adce8cee840eb202fbc9219069b50b40d5ada01` (1,592,102,392 B in CI, run `44455733670`,
  job `135330174857`, commit `ee072d84acacf5d64c6cff36dc611d7c9ed8a875`; 2.04 GB uncompressed on host), built from
  upstream `v0.2.16` commit `64524fb9fc82222ddce554c0643b64f056ec8366` (Terminal 0.2.16 / Control `0.1.62`). Engine
  package `strata-engine-0.1.62-linux-x86_64-hip-rx7900.tar.gz` (276,843,935 B, asset sha256 `b0e14bf8…`, manifest
  `b969c28214698147bba9a3c09fcdd8cd45ba9466de148263b0da5641ae692089`, 587 files, `variant rx7900`, archs
  gfx1100/1101/1102); hipBLASLt table unchanged (`1bb8cd2e…`, 26 rows, version `100200`). 0.2.16 is a full engine
  generation: the engines run **Strata's own code** (no ggml / llama.cpp matrix kernels) and the picture reader is
  Strata's own (`strata-vision`) too. Control and the engines share one number from 0.2.16 on (no 0.1.61).
- **Build & image changes**: upstream moved the top-level folders into `engine/ app/ terminal/ docs/ tools/`
  (`docker/` → `tools/docker/`, `packaging/` → `tools/packaging/`, `third_party/` → `engine/third_party/`); all
  workflow paths follow. The 0.2.15 libatomic verify patch is **gone**: the 0.1.62 packages bundle `libatomic.so.1`
  themselves (`tools/packaging/notices/libatomic.txt` → `licenses/libatomic.txt` in the package; CI's `binary_deps`
  verify stage passes without a patch). The gguf-py `--ctx` pre-stage stays (upstream `tools/docker/build.sh` still
  omits `engine/third_party/strata-gguf-py`), now staged at `$ctx/terminal-src/engine/third_party/`. The entrypoint
  fork is now **four** changes: upstream 0.2.16 forwards `DRAFTING` / `DRAFT_VOCAB` itself (fork change 5 retired) and
  gained the `python_usable()` probe (issue #4309, carried); the fork keeps `--no-api-key`, the keyless warning, the
  pack chat-template swap and the vision budget (upstream entrypoint sha256 `1150f912…`).
- **Engine & serve changes for this box**: #6032 R1-R6 tool-call parser round (damaged openers, `<arguments>`+JSON,
  XML element parameters, `<function=NAME` without `>`, Claude's `<invoke …>` dialect; recovery on by default) and
  #4686 linear streamed tool-argument parsing; #6011 (a Stop reaches every stage of a layer split, stale checkpoint
  parts cleared); #6055 follow-up (a request cut by a stop fails `503`, not "length" with 0 tokens); #5847/#5848
  conversation-cache checkpoint plans/device tags; #5697 RAM gate + SSD tier and #2253 (park eviction); #5924
  (`fit_max_tokens` never shortens below answer room); #4758/#4759/#5633 stream fixes; #5893 `/v1/models` reports
  `context_length` + `meta.n_ctx`. New keyless `/v1` foreign-Origin check (owner 11-16): an origin equal to the
  request's Host passes and other origins are judged against `allowed_hosts` (ports ignored), so
  `STRATA_ALLOWED_HOSTS=fedora-tuf` covers the llama-swap UI; live-verified (own origin 200, foreign origin 403).
  #6067 (RX 7000 Gyro trellis fused prompt kernels, default-on) is inert for Flash-Next IQ3_S; #5860's two-card
  pipelined-decode knobs (`STRATA_PIPELINE_HIT_MIN` / `_LAZY_CHAIN` / `_LOOKUP_RUN`) are off by default — an A/B
  candidate for later. DFlash2 still refuses beside `--pipeline-windows`, so the lanes stay MTP.
- **Rollout (2026-11-19)**: host checkout pulled `ee072d84` (image tag + fork + config). `docker pull` from ghcr.io
  fails on fedora-tuf (TLS EOF / SNI-level block); pulled through the local mihomo proxy with
  `HTTPS_PROXY=http://127.0.0.1:2080 skopeo copy docker://… docker-daemon:…`, after checking the registry digest
  (`6a4df9df…`, label 0.2.16) and the in-image inventory (`0.1.62 linux-x86_64 hip`, `usable: true`, no problems).
  One setup pass per lane (the `0.1.60` run settings name the old engine folder and are dropped; packs/MTP reused):
  cold ~60 s, warm 57 s on both lanes. The vision budget re-applied on the reload
  (`Vision budget: strata-iq3_s.json: max_tokens=8192, min_tokens=1024`); the reader is the new own tower
  (`strata-vision: own tower (qwen3vl_merger, 27 layers, width 1152) … on the GPU`; warm-up 0.7 s / 956 MiB at the
  1024 default, 10.2 s / 1,161 MiB at 4096 tokens with the 8192 cap). Placement: vanilla first start (setup default
  1024) `8,365 + 7,609 = 15,974`
  slots; warm (vision 8192) **`8,249 + 7,609 = 15,858`** (14.96 + 15.25 GiB), RAM copy `21.99 GiB`, lend `2,816`,
  MTP `1,348 MiB`, free 1,814/1,338 MiB; SC117 **`8,198 + 7,575 = 15,773`** (14.87 + 15.18 GiB), RAM copy `22.15 GiB`,
  lend `2,817`.
- **Verification**: 17/17 llama-swap gates on both lanes (reasoning → 391, `:npt` alias, declared `get_weather` with
  `include_forecast: true` as a JSON bool (#5979), multi-turn tool result, vision red/blue through the own reader,
  `SMOKE-OK`, `/props` n_ctx 262144, own-Origin 200 / foreign-Origin 403). In-image serve suite: **1,931 tests,
  11 errors — all environment-only** (repo-only files absent from the image: app `Monitor.tsx`, the
  `qwen4_intake_fixtures`, and the `web_i18n` / `make_profile` / `test_strata_tokenizer` modules), 88 skipped; the
  focused parser suites **124/124 OK** and `test_tool_format` **12/12 OK** with `terminal/tools/qwen4_intake_fixtures`
  mounted from the upstream v0.2.16 tree. Graceful stop 2.77 s (vanilla) / 3.84 s (SC117); no GPU-pause or engine
  error lines after the starts. Longctx spot matched to 0.2.14.1 (`--out-pct 10`, 60/85%, 2 reps):
  `2,592.9/66.90` and `2,660.7/64.15` vs `2,529.2/66.65` and `2,710.0/65.55` — prompts and decode neutral
  ([`qwen3.8/Flash-Next.md`](qwen3.8/Flash-Next.md) 20261119).

