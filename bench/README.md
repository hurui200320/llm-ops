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

```bash
npm install -g @syntheticlab/synbad
# synbad requires an env-var name for provider auth even against keyless
# endpoints; llama-swap runs with apiKeys: [], so a dummy value is fine.
export LLAMA_SWAP_KEY=dummy
synbad eval --env-var LLAMA_SWAP_KEY --base-url "http://fedora-tuf:8080/v1" --model "<alias>" --count 100
synbad eval --env-var LLAMA_SWAP_KEY --base-url "http://fedora-tuf:8080/v1" --model "<alias>" --count 100 --stream
```

`--count 100` is deliberate: synbad's README notes the response-in-reasoning
bug only reproduces reliably at 40+ runs, and against a local endpoint the only
cost is time.

### 1. Speed (bench/speed/)

Two tools, both through llama-swap so they measure the deployed config itself
(chat template, ctx-checkpoints, cache, speculative decoding):

- [`run_speed_bench.sh`](speed/run_speed_bench.sh) — NVIDIA SPEED-Bench client
  (fetched from llama.cpp on the fly): realistic short/medium prompts
  (coding/roleplay/…), per-category pp/tg t/s + draft acceptance, and
  baseline-vs-spec comparison. Requests keep the server's normal prompt
  caching (the client pins temperature 0), so multi-turn samples reuse the
  conversation prefix like real chat traffic. Because a model kept loaded
  holds its prompt cache and ctx checkpoints, **unload all models on
  llama-swap before every run** (`curl http://fedora-tuf:8080/unload`, or
  the UI) — an accidental cache hit would fake a fast result. Throughput
  splits (fixed 1k–32k ISL) are available through its arg passthrough, e.g.
  `./run_speed_bench.sh run <alias> <tag> --bench throughput_32k --category mixed`.
- [`run_longctx.py`](speed/run_longctx.py) — the near-full-context worst case:
  85% of the max context of meaningful text in, 10% out — the shape of a
  harness compaction request. Cold prefill per rep (distinct corpus segments,
  `cache_prompt: false`), reports pp/tg t/s and spec accept rate at depth.

```bash
cd speed
./run_speed_bench.sh run ornith-1.5-35b-a3b-q80-vision ornith
python3 run_longctx.py --model ornith-1.5-35b-a3b-q80-vision                       # 256K ctx
python3 run_longctx.py --model meta-muse-glimmer-30b-kquant-vision --ctx 131072   # 128K per slot
```

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
tool-eval-bench probe --base-url http://fedora-tuf:8080
TOOL_EVAL_MODEL=<alias> tool-eval-bench run --base-url http://fedora-tuf:8080 --seed 42
# later: tool-eval-bench compare <runA> <runB>
```

Run it from `bench/` so its `runs/` + `data/` artifacts (written relative to the
CWD) land in gitignored dirs.

Note: its safety category penalizes uncensored models — interpret abliterated
variants' scores with that in mind.

### 3. Reasoning (bench/reasoning/)

Small deterministic suite: 24 light (GSM8K-style) + 12 hard (AMC/AIME-style)
problems, exact-match scoring, pinned temperature. Also reports per-request tg
speed as a realistic reasoning-workload number.

```bash
cd reasoning
python3 run_reasoning.py --model <alias>
python3 run_reasoning.py --model <alias> --difficulty hard   # subset
```

Add problems to `problems.jsonl`; answers must be independently verified. Note
the suite is a smoke check: with 24+12 problems one answer swings accuracy by
~4/8pp, so don't use it alone to break ties between close models.

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
