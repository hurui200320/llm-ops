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
  size. Defaults to a four-level curve — 20%, 40%, 60%, 85% of the max context
  of meaningful text in, 10% of max context out at every level — the shape of
  a harness compaction request, worst case last. The max context is
  auto-detected from llama-server `/props` through the
  llama-swap upstream passthrough (per-slot `n_ctx` when parallel slots are
  configured — the cap for a single request); `--ctx N` overrides it with a
  warning on mismatch and is needed for non-llama.cpp backends. Other sizes:
  `--in-pcts 20,40,60,85 --out-pct 10`. Cold prefill per rep
  (distinct corpus segments within a level, `cache_prompt: false` across levels
  verified by `cache_n == 0`), so no unload-before-run dance is needed;
  reports per-level pp/tg t/s and spec accept rate at depth.

```bash
cd speed
python3 run_longctx.py --model ornith-1.5-35b-a3b-q80-vision                      # ctx auto-detected from /props
python3 run_longctx.py --model meta-muse-glimmer-30b-kquant-vision               # 128K per slot, also auto-detected
python3 run_longctx.py --model <alias> --ctx 262144                              # override (warns if /props disagrees)
python3 run_longctx.py --model <alias> --in-pcts 20,40,60,85 --out-pct 10        # explicit (these are the defaults)
```

Wrapped up as [bench/run_longctx.sh](../bench/run_longctx.sh): for every model
in `deploy/llama-swap.config.yaml` (positional args subset it) runs
`run_longctx.py --reps 2` (no ctx override — auto-detected), teeing each run to
`bench/results/longctx/<model>-<stamp>.log` with the results JSON next to it.
The JSON holds one entry per input level (`levels[]`, each with its own
`summary` + `results`); the top-level `summary` aggregates across levels
(embedding a `levels[]` list that mirrors each level's summary).
Env overrides: `LLAMA_SWAP_URL`, `REPS`, `IN_PCTS`, `OUT_PCT`.

Expect tens of minutes per `run_longctx.py` rep at the largest level: a cold
prefill plus a long decode, times levels x reps (default 4 levels x 2 reps).

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

### 3. Reasoning gate (bench/reasoning/)

Pass/fail gate: 1 sanity problem + 3 AIME 2026 + 6 zebra logic-grid puzzles
(`run_reasoning.sh` selects these 10 via `SKIP_IDS` out of 43 on disk:
12 light sanity-math in problems.jsonl, 15 AIME 2026 problems and
16 zebra puzzles built by `fetch_frontier.py` into
`../.cache/reasoning/frontier.jsonl`). Numeric problems are exact-match
scored; zebra puzzles are scored puzzle-level (full grid match) plus
cell-level partial credit, and per-request completion-token counts are
recorded. Sampling is left at the server default on purpose: this measures
the models under the deployed setup, not a leaderboard number.

Gate set: sanity `l12` (must pass — a fail means the harness is broken, not
the model); AIME `aime26-10, aime26-11, aime26-15`; zebra `5x5:3, 6x4:2,
6x6:1`. PASS = `l12` ok and >=6/9 scored; REVIEW = 5/9; FAIL = <=4/9 or
`l12` fails. Zebra carries the weight (cell credit shows defect shape;
AIME is binary), and AIME is only 3 problems so one lucky guess can't pass
the gate. `aime26-15` (0/8 in Sep 2026) and `zebra-6x6-01` are
near-impossible anchors: if a future model solves them, its headroom shows
up here. Retries are a single 60s wait (`RETRY_WAITS_S = (60,)` in
`run_reasoning.py`): a problem burning the full 30-minute `--timeout`
twice is a genuine fail, and on a single-slot setup each extra retry is
another half hour of wall. Expect ~45-90 min per model.

Comparison runs no longer try to rank: `SKIP_IDS=` runs everything (43),
only useful for calibrating a new gate set. The frozen manifest and the
problem files always hold the full set. Transient gateway 5xx — what a
restarting llama-swap answers instantly — get the single retry above;
4xx stays fatal and the run aborts after 5 consecutive errors, so
mid-run restarts no longer silently lose problems.

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

**Everything passes** (ceiling): move the zebra mix up the ladder — e.g.
`--zebra-spec "4x5:4,5x5:5,6x4:4,6x6:2"` — and take `--aime-count 30`.
If the strongest model still sweeps 6x6, this generator is exhausted:
put more counts on 6x6 for statistical power and read the cell-credit
and completion-token columns instead. If AIME passes with suspiciously
short thinking, suspect training contamination (the exam dates to Feb 2026)
and weight zebra (generated, never seen) more heavily.

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

Resolution honesty: ~27 frontier problems per comparison run means one problem
swings a family score by ~7pp. Call a difference real only when it shows up in
per-problem flips, zebra cell credit or completion tokens too, not the
aggregate alone.

### 4. Agentic coding (mini-SWE-agent on SWE-bench Multilingual)

The model gets a GitHub issue + repo checkout in Docker and must drive the
loop itself — read/grep/edit via bash tool calls, then submit a patch scored
by the repo's own tests. This is the same mechanism as OpenCode, unlike the
old aider-polyglot layer (which injected files and parsed edit blocks, no
tool calls — deleted as uninformative for this use case).

Dataset: [SWE-bench Multilingual](https://www.swebench.com/multilingual.html)
(300 tasks, native SWE-bench format). No Kotlin exists in any standard agent
benchmark, so a frozen 20-instance pilot (8 java + 8 js/ts + 4 cpp) stands in,
with Java as the JVM/Gradle/JUnit proxy for Kotlin. Repo picks avoid heavy
builds (`logstash`, `druid`) and long eval scripts; instance_ids are frozen
in the runner — reruns on the same filter are comparable, different filters
are not. Expand only when several models saturate this set (all correct).

Wrapped up as [bench/run_agentic.sh](run_agentic.sh): installs mini-swe-agent
and the `swebench` eval package via uv on the fly (never vendored; the eval
runs as `uvx --from swebench` because a uv-tool venv isn't importable by
system python3), renders [bench/agentic-swebench.yaml](agentic-swebench.yaml)
per model (`openai/<alias>` → `$LLAMA_SWAP_URL/v1`, server-side sampling,
sequential tool calls, cost tracking off), then for every model in
`deploy/llama-swap.config.yaml` (positional args subset it) runs
`mini-extra swebench --workers 1` with the frozen `--filter`, followed by the
local eval harness on `preds.json` inside the run dir. Per-model output lands
in `bench/results/agentic/<model>-<stamp>/` (gitignored): `preds.json` (the
graded copy; original kept as `preds.raw.json`), per-instance trajectory
dirs, the rendered `agent-config.yaml`, teed `console.log` + `eval.log`, the
harness `eval-report.json` copy, and a `verdict.txt` with resolved/total
overall + per language. The agent's "Submitted" status only means a patch was
produced — `verdict.txt` is the pass/fail answer. Env overrides:
`LLAMA_SWAP_URL`, `FILTER`, `CONFIG_TPL`, `SKIP_EVAL=1` (agent only),
`EVAL_ONLY=<run-dir>` (re-grade an existing `preds.json` without redoing the
agent loop), `EVAL_TIMEOUT` (per-instance test seconds, default 1800).

```bash
./bench/run_agentic.sh                         # all models, frozen 20-instance pilot
./bench/run_agentic.sh <alias>                 # one model (smoke first)
FILTER='^(axios__axios-4738)$' ./bench/run_agentic.sh <alias>   # 1-instance smoke
```

One worker: the servers run `--parallel 1`. Expect ~3–10 min/instance on
JS/TS and ~5–18 min on Java/C++ (Maven/Gradle + Docker exec dominate), so
~3–4h per model on the pilot. Docker required; LLM-written code is executed
unsupervised, so keep it in the container.

Agent loop settings live in `agentic-swebench.yaml` (pinned copy of
upstream's prompts; local deviations marked `[LOCAL]`): `step_limit: 75`
(down from upstream 250 — bounds worst-case wall on slow local inference),
env `timeout: 600` (Maven/Gradle exceed upstream's 60s), sequential tool
calls (single-request serving + Glimmer one-tool-per-turn). Pin these across
compared models, or don't compare.

`verdict.txt` answers "which model is best at which language" (resolved/total
overall + per language + id lists); record it in `notes/bench.md`.

Reading the verdict:
+ `resolved` = F2P+P2P all passed, the model earns credit;
+ `unresolved` = grading ran but the answer isn't accepted, no credit.
  These are model scores, not harness failures.
+ `errors`/`empty`/`incomplete` = harness failed to grade due to crashes/empty
  patches land in.

Caveat: `unresolved` lumps "wrong code" with "right code rejected on a technicality"
(observed: a functionally correct axios patch scored unresolved because the instance's
`timeout 10s ... mocha` wrapper exits 124 while mocha reports 4/4 pass, tripping the
harness exit-code guard — plus the patch missing its trailing newline, forcing a
`--reject` partial apply). For comparing models use `unresolved` as-is; for
diagnosing one instance read its per-instance `report.json` + `test_output.txt`
under `logs/run_evaluation/<run_id>/`.

Reviewing failures — don't stop at x/y (`verdict.txt` is the scoreboard,
not the diagnosis):
+ `Submitted` + non-empty patch in `preds.json` = capability signal, graded
  by the repo tests into `resolved`/`unresolved`.
+ Empty patch + trajectory `exit_status: LimitsExceeded` = never submitted
  (budget/process failure), not a wrong-answer verdict. The `step_limit` cap
  measures efficiency; a context overflow would instead show as API/context
  errors or `finish_reason: length` truncation.
+ Triage cheap first: patch size in `preds.json` plus `exit_status` /
  `api_calls` in `<instance>/<instance>.traj.json`. Then read the tail tool
  commands to split capped runs into progressing (diagnosed the bug, fumbling
  submission/scaffolding) vs. thrashing (repeated unproductive edits, no
  source fix).
+ Only capped-but-progressing instances justify an extended-budget re-run:
  `FILTER='^(<instance>)$' ./bench/run_agentic.sh <alias>` with a raised
  `step_limit`, reported as a separate labeled column. Never compare models
  across different budgets. Record the per-instance why (no-submit vs.
  wrong-fix vs. harness-reject) in `notes/bench.md` next to the counts.

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
