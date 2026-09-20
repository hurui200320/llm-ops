# bench

Personal benchmark pipeline for models deployed on the llama-swap + llama.cpp
setup (`deploy/`). Answers three questions per model: 

+ does it work on my stack
+ how fast (or slow) is it
+ is it better than the alternatives for my use cases

This is not a leaderboard. Results go into `notes/bench.md`; raw
artifacts land in `bench/results/` (gitignored).

Everything assumes llama-swap is reachable at `http://fedora-tuf:8080` and
speaks OpenAI-compatible API with the model alias as `model`. Each tool
documents its own overrides: the Python scripts take flags (e.g. `--base-url`),
the shell scripts read env vars (e.g. `LLAMA_SWAP_URL`, `LIMIT`).

## Pipeline per model

Run layers in this order; each is standalone.

> Note: since llamacpp and other software changes fast, ideally we should
> rerun the tests for ALL models using the same version of software.

### 0. Gate — does the stack work? (synbad)

[Synbad](https://github.com/synthetic-lab/synbad) detects *serving stack* bugs
(tool-call parsing, parallel tool calls, reasoning-field parsing, streaming vs
non-streaming). It does not measure model intelligence. If this fails, fix the
chat template / server flags first; capability numbers on a broken stack are
meaningless.

Wrapped up as [bench/run_synbad.sh](../bench/run_synbad.sh): installs/updates
synbad via npm on the fly, then for every model in `deploy/llama-swap.config.yaml`
(positional args subset it) runs `synbad eval --count 20` with and without
`--stream`. Logs land in `bench/results/synbad/`; env overrides: `LLAMA_SWAP_URL`,
`COUNT`, `LLAMA_SWAP_KEY`.

Or manually:

```bash
npm install -g @syntheticlab/synbad
# synbad requires an env-var name for provider auth even against keyless
# endpoints; llama-swap runs with apiKeys: [], so a dummy value is fine.
export LLAMA_SWAP_KEY=dummy
synbad eval --env-var LLAMA_SWAP_KEY --base-url "http://fedora-tuf:8080/v1" --model "<alias>" --count 10
synbad eval --env-var LLAMA_SWAP_KEY --base-url "http://fedora-tuf:8080/v1" --model "<alias>" --count 10 --stream
```

`--count 10` (per mode, 10+10 total) keeps the gate fast on big models. Synbad's
README says 40 per mode is typically good enough to reveal the bugs — in
particular the 5%-rate response-in-reasoning one — so re-run with `COUNT=40`
if a failure looks borderline or flaky.

### 1. Speed (bench/speed/)

One tool, through llama-swap so it measures the deployed config itself
(chat template, ctx-checkpoints, cache, speculative decoding):

- [`run_longctx.py`](speed/run_longctx.py) — generation speed vs. context
  size. Defaults to the near-full-context worst case: 85% of the max context
  of meaningful text in, 10% out — the shape of a harness compaction request.
  The max context is auto-detected from llama-server `/props` through the
  llama-swap upstream passthrough (per-slot `n_ctx` when parallel slots are
  configured — the cap for a single request); `--ctx N` overrides it with a
  warning on mismatch and is needed for non-llama.cpp backends. Other sizes:
  rerun with `--isl N --osl M` overrides. Cold prefill per rep
  (distinct corpus segments, `cache_prompt: false`), so no unload-before-run
  dance is needed; reports pp/tg t/s and spec accept rate at depth.

```bash
cd speed
python3 run_longctx.py --model ornith-1.5-35b-a3b-q80-vision                      # ctx auto-detected from /props
python3 run_longctx.py --model meta-muse-glimmer-30b-kquant-vision               # 128K per slot, also auto-detected
python3 run_longctx.py --model <alias> --ctx 262144                              # override (warns if /props disagrees)
```

Wrapped up as [bench/run_longctx.sh](../bench/run_longctx.sh): for every model
in `deploy/llama-swap.config.yaml` (positional args subset it) runs
`run_longctx.py --reps 2` (no ctx override — auto-detected), teeing each run to
`bench/results/longctx/<model>-<stamp>.log` with the results JSON next to it.
Env overrides: `LLAMA_SWAP_URL`, `REPS`.

Expect tens of minutes per `run_longctx.py` rep: a cold ~full-context prefill
plus a long decode.

Deployment heuristic — speed is mostly decided by whether things fit in VRAM:

1. Pick the biggest quant that fits **with real headroom**. When VRAM is merely
   "enough to load" (e.g. 23.1/23.9 GB), llama-server ends up fighting the
   desktop stack (gnome shell etc.) for VRAM and everything silently slows
   down; observed on gemma-4-31b-qat and ornith-1.5-35b.
2. Start with MTP enabled, run the speed tests while watching VRAM (nvtop /
   amdgpu-top). If usage is tight, drop the MTP — it is unstable and risky at
   that point. The MTP head is under ~1 GB for 35B models, so dropping it
   never bumps the main model a quant level; the decision is purely about
   headroom.
3. Rerun the tests after dropping MTP and compare the saved JSON runs to see
   what it was actually costing.

Draft acceptance only means something on meaningful input, which is why
`run_longctx.py` feeds a real public-domain novel (Project Gutenberg, fetched
on the fly, cached under `bench/.cache/`) instead of synthetic filler.

### 2. Tool calling (tool-eval-bench)

[tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench): 69
deterministic scenarios (tool selection, parameter precision, chains, error
recovery, structured output) through `/v1/chat/completions`. Its mock tools
exercise the same schema path EXA-MCP traffic uses.

```bash
uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git
```

Wrapped up as [bench/run_tooleval.sh](run_tooleval.sh): for every model in
`deploy/llama-swap.config.yaml` (positional args subset it) runs
`tool-eval-bench run --seed 42`, cd'ing into `bench/` so its `runs/` + `data/`
artifacts (written relative to the CWD) land in gitignored dirs. Each run is
teed to `bench/results/tooleval/<model>-<stamp>.log`. Env overrides:
`LLAMA_SWAP_URL`, `SEED`.

Later, compare runs side by side: `tool-eval-bench compare <runA> <runB>`.

Note: its safety category penalizes uncensored models — interpret abliterated
variants' scores with that in mind.

### 3. Reasoning (bench/reasoning/)

Three families, ~42 problems: 12 light sanity-math (problems.jsonl), 15 AIME
2026 problems and 15 zebra logic-grid puzzles (both built by
`fetch_frontier.py` into `../.cache/reasoning/frontier.jsonl`). Numeric
problems are exact-match scored; zebra puzzles are scored puzzle-level (full
grid match) plus cell-level partial credit, and per-request completion-token
counts are recorded (verbosity drift is a secondary quant signal). Sampling is
left at the server default on purpose: this measures the models under the
deployed setup, not a leaderboard number.

Why the tier mix: the old light+hard split saturated (Gemma 4 26B scored
36/36, all `finish: stop`, nowhere to differentiate quants or abliterated
variants). AIME 2026 is fresh competition math; zebra puzzles are long pure
deduction where one bad step cascades — both sit at the frontier where
strong models land in the 50-90% band. Zebra generation follows the
ZebraLogicBench recipe (random solution grid, all true clues, minimize under
a uniqueness-checking CSP solver) because its public release hides the
answers; being generated also makes them contamination-free.

```bash
cd reasoning
python3 fetch_frontier.py                       # build frontier.jsonl + freeze the manifest (once)
python3 fetch_frontier.py --check               # verify cache still matches the frozen manifest
python3 run_reasoning.py --model <alias> \
    --problems problems.jsonl ../.cache/reasoning/frontier.jsonl --max-tokens 65536
python3 run_reasoning.py --model <alias> --family zebra    # subset by family
```

#### Building the frontier tier (`fetch_frontier.py`)

One invocation builds the whole tier and freezes it: AIME 2026 rows are
fetched from `MathArena/aime_2026` (revision-pinned; MAA-copyrighted text
stays in the gitignored cache), zebra puzzles are generated from the seed.
`frontier_manifest.json` (committed) pins ids + answer hashes — no problem
text is committed. `--check` verifies the cache still matches the frozen
manifest, and `run_reasoning.sh` refuses to run on drift.

Two difficulty dials:

- `--aime-count` (default 15, max 30): problems are taken in fixed order;
  15 is one exam's worth, 30 both. Difficulty is fixed by the source — the
  only dial is coverage.
- `--zebra-spec` (default `3x4:2,4x4:5,4x5:3,5x5:3,6x4:2`): `NxM:count`
  entries. N = houses, M = characteristics; difficulty tracks the search
  space (N!)^M, which is *not* the same order as the N*M product:

  | size | search space |  | size | search space |
  |------|--------------|--|------|--------------|
  | 3x4  | ~1.3e3       |  | 5x5  | ~2.5e10      |
  | 4x4  | ~3.3e5       |  | 6x4  | ~2.7e11      |
  | 4x5  | ~8e6         |  | 6x6  | ~1.4e17      |

  6x4 out-searches 5x5 despite the smaller product, and 6x6 is a category
  of its own. `--seed` only swaps in fresh puzzles (new manifest); it is
  not a difficulty knob.

**Everything passes** (ceiling, like the old 36/36 run): move the zebra mix
up the ladder — e.g. `--zebra-spec "4x5:4,5x5:5,6x4:4,6x6:2"` — and take
`--aime-count 30`. If the strongest model still sweeps 6x6, this generator
is exhausted: put more counts on 6x6 for statistical power and read the
cell-credit and completion-token columns instead. If AIME passes with
suspiciously short thinking, suspect training contamination (the exam dates
to Feb 2026) and weight zebra (generated, never seen) more heavily.

**Everything fails** (floor): before blaming difficulty, triage the harness —
`finish_reason: "length"` means raise `MAX_TOKENS`/`--max-tokens` (zebra
puzzles think long: even a 3x4 costs ~5k completion tokens), and replies
without a `SOLUTION:` block mean format non-compliance — which is itself a
model difference; check the raw `reply` field in the results JSON first. If
the tier is genuinely too hard, shift the spec down (e.g.
`3x4:4,4x4:6,4x5:4,5x5:2`) — the small bands exist so weak models have
somewhere to land.

**Target picture**: the strongest model around 50-90% per family, weaker
models spreading below it. Any argument change re-freezes the manifest, and
results across different manifests are not comparable — calibrate on the
strongest model first, freeze once, then run the whole lineup paired.

Resolution honesty: ~30 frontier problems means one problem swings a family
score by 3-7pp. Call a difference real only when it shows up in per-problem
flips, zebra cell credit or completion tokens too, not the aggregate alone.

### 4. Agentic coding (aider polyglot)

The [aider polyglot benchmark](https://github.com/Aider-AI/aider/tree/main/benchmark)
runs in Docker against the OpenAI-compatible endpoint; LLM-written code is
executed unsupervised, so keep it in the container. Subset to the languages I
use (java / javascript / python — no kotlin/ts exist in polyglot).

```bash
git clone https://github.com/Aider-AI/aider.git && cd aider
git clone https://github.com/Aider-AI/polyglot-benchmark tmp.benchmarks/polyglot-benchmark
docker build -t aider-benchmark -f benchmark/Dockerfile .
docker run --rm -it -e AIDER_DOCKER=1 \
    -e OPENAI_API_BASE=http://host.docker.internal:8080/v1 -e OPENAI_API_KEY=dummy \
    --add-host=host.docker.internal:host-gateway \
    -v "$PWD:/aider" -w /aider aider-benchmark bash
# inside:
pip install -e '.[dev]'
cat > .model-settings.yml <<'EOF'
- name: openai/<alias>
  edit_format: whole
  weak_model_name: openai/<alias>
  use_temperature: false   # let llama-server sampling apply
EOF
./benchmark/benchmark.py <run-name> --model openai/<alias> \
    --edit-format whole --threads 1 --keywords java,python \
    --read-model-settings .model-settings.yml --exercises-dir polyglot-benchmark
```

(`--keywords java` also matches javascript.) One thread: the servers run
`--parallel 1`. Expect hours; start with `--num-tests 2` to smoke-test.

### 5. Chinese RP (bench/rp/)

Scaffold: fixed Chinese scenarios → generate continuations via the same
endpoint SillyTavern uses → judge with a pinned online model (ZenMux). See
[rp/README.md](rp/README.md). Needs real scenarios before it means anything.

> Current status: WIP, as I don't have any good examples to test

## Conventions

- Python scripts here are standard-library only (same discipline as `scripts/`,
  though the toolbox policy does not cover `bench/`).
- No secrets in the repo: judge key via `ZENMUX_API_KEY`, endpoints are LAN.
- `bench/results/`, `bench/.cache/`, `bench/runs/` and `bench/data/` are gitignored.
- Pin judge model / seeds / sampling across compared models, or don't compare.
