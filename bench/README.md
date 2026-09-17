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
the shell scripts read env vars (e.g. `LLAMA_SWAP_URL`, `MODELS_DIR`).

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

Two complementary tools:

- [`run_bench.sh`](speed/run_bench.sh) — llama-bench in the deployment docker
  image: pp/tg at context depths 0 → 90% of max context (worst case;
  muse-glimmer capped at its 128K per-slot context), per-model
  flags from [`models.conf`](speed/models.conf). Cannot measure speculative decoding.
- [`run_speed_bench.sh`](speed/run_speed_bench.sh) — NVIDIA SPEED-Bench client
  (fetched from llama.cpp on the fly) against llama-swap: realistic prompts
  (coding/roleplay/…), per-category t/s + draft acceptance, and
  baseline-vs-spec comparison to decide if MTP/DFlash is worth the VRAM.

```bash
cd speed
./run_bench.sh --list
./run_bench.sh ornith15-35b-a3b-q80
./run_speed_bench.sh run ornith-nospec ornith-nospec
./run_speed_bench.sh run ornith-mtp    ornith-mtp
./run_speed_bench.sh compare ../results/speed/speed-bench-ornith-nospec.json \
                              ../results/speed/speed-bench-ornith-mtp.json
```

A spec A/B needs two llama-swap aliases for the same model (one
`--spec-type none`, one with spec enabled).

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
