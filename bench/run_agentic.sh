#!/usr/bin/env bash
# Agentic coding runner (bench step 4): runs a frozen 20-instance pilot slice
# of SWE-bench Multilingual through mini-SWE-agent in batch mode, pointed at
# the llama-swap OpenAI-compatible endpoint. The model gets a GitHub issue +
# repo checkout in Docker and must drive the loop itself (bash/file tools);
# pass/fail comes from the repo's own tests. This replaces the old
# aider-polyglot layer, which injected files and parsed edit blocks — no tool
# calls, so it measured the wrong thing for OpenCode-style use.
#
# mini-swe-agent is installed on the fly via uv (never vendored); per-instance
# SWE images are pulled by docker on first use. LLM-written code is executed
# unsupervised, so it stays in the container.
#
# Usage:
#   ./run_agentic.sh                      # all models from deploy/llama-swap.config.yaml
#   ./run_agentic.sh <alias> [alias...]   # only the named llama-swap models
#
# Each model gets results/agentic/<model>-<stamp>/ with preds.json
# (sanitized copy graded; original kept as preds.raw.json), per-instance
# trajectory dirs, the rendered agent-config.yaml, teed console.log +
# eval.log, the harness eval-report.json copy and a verdict.txt summary —
# all gitignored. Pass/fail comes from the local SWE-bench eval harness
# (run via `uvx --from swebench`, since a `uv tool install` venv is not
# importable by system python3), which applies each model_patch and runs the
# repo's own tests in Docker; the agent's "Submitted" status alone says
# nothing about correctness. Eval runs with CWD inside the run dir, so the
# harness's relative logs/run_evaluation/ tree lands there too.
#
# Env:
#   LLAMA_SWAP_URL  llama-swap root URL (default http://10.233.1.16:8080; /v1
#                   is appended for the agent config)
#   FILTER          --filter regex pinning the instance set (default: frozen
#                   20-instance pilot below — 8 java + 8 js/ts + 4 cpp; reruns
#                   on the same filter are comparable, different filters are not)
#   CONFIG_TPL      agent config template (default bench/agentic-swebench.yaml)
#   SKIP_EVAL       set to 1 to skip the eval step (agent run only)
#   EVAL_ONLY       set to an existing results/agentic/<model>-<stamp> dir to
#                   skip the agent run and only (re-)grade its preds.json
#   EVAL_TIMEOUT    per-instance test timeout in seconds (default 1800)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/../deploy/llama-swap.config.yaml"
CONFIG_TPL="${CONFIG_TPL:-$SCRIPT_DIR/agentic-swebench.yaml}"
RESULTS_DIR="$SCRIPT_DIR/results/agentic"

LLAMA_SWAP_URL="${LLAMA_SWAP_URL:-http://10.233.1.16:8080}"
BASE_URL="$LLAMA_SWAP_URL/v1"
STAMP="$(date +%Y%m%d-%H%M%S)"

# Frozen pilot: 8 java + 8 js/ts + 4 cpp from SWE-bench Multilingual.
# Repo picks avoid the heavy builds (logstash, druid) and the long eval
# scripts (lombok-3042/3052); instance_ids verified against the dataset.
# Expand only when several models saturate this set (see bench/README.md).
FILTER="${FILTER:-^(google__gson-1093|google__gson-1100|google__gson-2311|projectlombok__lombok-3312|projectlombok__lombok-3326|projectlombok__lombok-3479|apache__lucene-12196|apache__lucene-13170|axios__axios-4738|axios__axios-6539|vuejs__core-11739|vuejs__core-11870|preactjs__preact-3454|preactjs__preact-4316|mrdoob__three.js-25687|mrdoob__three.js-27395|fmtlib__fmt-1683|fmtlib__fmt-2457|fmtlib__fmt-3272|nlohmann__json-4237)$}"

# llama-swap is LAN; a shell-wide proxy (e.g. ALL_PROXY=socks://) must never
# apply here (same reason as run_tooleval.sh / run_reasoning.sh)
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

CYAN=$'\033[96m'
RESET=$'\033[0m'

ensure_agent() {
    if command -v mini-extra >/dev/null 2>&1; then
        return
    fi
    echo "mini-extra not on PATH, installing mini-swe-agent via uv..." >&2
    uv tool install mini-swe-agent || {
        echo "error: uv tool install mini-swe-agent failed and mini-extra is not on PATH" >&2
        exit 1
    }
}

ensure_eval() {
    # the local eval harness grades preds.json by applying each model_patch
    # and running the repo tests in Docker — this is what decides pass/fail.
    # Installed as an isolated uv tool venv (like mini-swe-agent): not
    # importable by system python3, so always invoked as
    # `uvx --from swebench python -m swebench.harness.run_evaluation`.
    if uvx --from swebench python -c "import swebench.harness.run_evaluation" >/dev/null 2>&1; then
        return
    fi
    echo "swebench not available via uvx, installing via uv..." >&2
    uv tool install swebench || {
        echo "error: uv tool install swebench failed" >&2
        exit 1
    }
}

models_from_config() {
    yq '.models | keys | .[]' "$CONFIG"
}

warmup() {
    # llama-swap loads a model on its first request; trigger the load here so
    # a slow cold start can't be misread as an agent failure
    local model="$1"
    curl -sf --max-time 600 "$BASE_URL/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
        >/dev/null || echo "warning: warmup request for $model failed, continuing anyway" >&2
}

sanitize_preds() {
    # Defense-in-depth: strip any BASH_ENV/conda.sh noise line that leaks into
    # captured command output (the frozen pilot no longer sets BASH_ENV, but an
    # old preds.json or a future Python-image instance could still carry it).
    # git apply would choke on it, so grade a sanitized copy and keep the
    # original untouched as preds.raw.json. Prints the sanitized path.
    local outdir="$1"
    [[ -f "$outdir/preds.raw.json" ]] || cp "$outdir/preds.json" "$outdir/preds.raw.json"
    python3 - "$outdir/preds.raw.json" "$outdir/preds.json" <<'EOF'
import json, sys
raw = json.load(open(sys.argv[1]))
clean = {k: {**v, "model_patch": "\n".join(
    ln for ln in (v.get("model_patch") or "").splitlines()
    if "conda.sh: No such file or directory" not in ln)} for k, v in raw.items()}
json.dump(clean, open(sys.argv[2], "w"), indent=2)
EOF
    echo "$outdir/preds.json"
}

run_eval() {
    # Grade OUTDIR/preds.json in Docker; writes eval.log, eval-report.json and
    # verdict.txt into OUTDIR. Runs with CWD inside OUTDIR so the harness's
    # relative logs/run_evaluation/ tree is self-contained there.
    local model="$1" outdir="$2"
    local eval_id="agentic-$(echo "$outdir" | tr '/:' '__')"
    echo "${CYAN}  evaluating preds.json (run_id=$eval_id)...${RESET}"
    local preds
    preds="$(sanitize_preds "$outdir")"
    local eval_ok=1
    if (cd "$outdir" && uvx --from swebench python -m swebench.harness.run_evaluation \
        --dataset_name SWE-bench/SWE-bench_Multilingual --split test \
        --predictions_path "$(basename "$preds")" \
        --max_workers 1 --timeout "${EVAL_TIMEOUT:-1800}" \
        --run_id "$eval_id" 2>&1 | tee eval.log); then
        echo "${CYAN}  eval done${RESET}"
    else
        echo "warning: model $model eval FAILED, log: $outdir/eval.log" >&2
        FAILED+=("$model (eval)")
        eval_ok=0
    fi
    summarize_verdict "$model" "$outdir" "$eval_id" || \
        echo "warning: verdict summary failed for $model" >&2
    return $eval_ok
}

summarize_verdict() {
    # Copy the eval harness report next to the run artifacts and write a
    # human-readable verdict: per-language resolved/total plus the id lists.
    # run_eval runs the harness with CWD inside OUTDIR, so its relative
    # logs/run_evaluation/ tree lands there. The harness summary report is
    # written to CWD (via --report_dir, default "."), named
    # <model_name_or_path with / as __>.<run_id>.json. Keep a copy as
    # eval-report.json plus the verdict at OUTDIR top level so each
    # model+stamp dir is self-contained.
    local model="$1" outdir="$2" eval_id="$3"
    local report
    report="$(ls "$outdir"/*."$eval_id".json 2>/dev/null | head -1)"
    [[ -n "$report" ]] || { echo "warning: eval report not found in $outdir (*.$eval_id.json)" >&2; return 1; }
    [[ -n "$report" ]] || { echo "warning: eval report not found in $outdir (*.$eval_id.json)" >&2; return 1; }
    cp "$report" "$outdir/eval-report.json"
    python3 - "$outdir/eval-report.json" "$outdir/verdict.txt" <<'EOF'
import json, sys
from collections import Counter
report_path, verdict_path = sys.argv[1], sys.argv[2]
r = json.load(open(report_path))
resolved = set(r.get("resolved_ids", []))
unresolved = set(r.get("unresolved_ids", []))
error = set(r.get("error_ids", []))
empty = set(r.get("empty_patch_ids", []))
infra = set(r.get("infra_failure_ids", []))
def lang(iid):
    return {"google": "java", "projectlombok": "java", "apache": "java",
            "axios": "js/ts", "vuejs": "js/ts", "preactjs": "js/ts",
            "mrdoob": "js/ts", "fmtlib": "cpp", "nlohmann": "cpp"}.get(iid.split("__")[0], "?")
lines = []
total = len(resolved | unresolved | error | empty)
lines.append(f"resolved {len(resolved)}/{total}")
by_lang = Counter()
for iid in resolved | unresolved | error | empty:
    by_lang[lang(iid)] += 1
for iid in resolved:
    by_lang[lang(iid) + "+ok"] += 1
for l in ("java", "js/ts", "cpp"):
    lines.append(f"  {l}: {by_lang[l + '+ok']}/{by_lang[l]}")
if resolved:
    lines.append("resolved: " + " ".join(sorted(resolved)))
if unresolved:
    lines.append("unresolved: " + " ".join(sorted(unresolved)))
if error:
    lines.append("errors: " + " ".join(sorted(error)))
if empty:
    lines.append("empty patch: " + " ".join(sorted(empty)))
if infra:
    lines.append("likely infra failures: " + " ".join(sorted(infra)))
open(verdict_path, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
}

ensure_agent
ensure_eval
[[ -f "$CONFIG_TPL" ]] || { echo "error: agent config template not found: $CONFIG_TPL" >&2; exit 1; }

if [[ -n "${EVAL_ONLY:-}" ]]; then
    # Re-grade an existing run dir without redoing the agent loop, e.g. to
    # grade the smoke run that predates the eval step:
    #   EVAL_ONLY=bench/results/agentic/<model>-<stamp> ./bench/run_agentic.sh
    [[ -d "$EVAL_ONLY" ]] || { echo "error: EVAL_ONLY dir not found: $EVAL_ONLY" >&2; exit 1; }
    [[ -f "$EVAL_ONLY/preds.json" ]] || { echo "error: no preds.json in $EVAL_ONLY" >&2; exit 1; }
    echo "${CYAN}Eval-only mode on $EVAL_ONLY${RESET}"
    run_eval "$(basename "$EVAL_ONLY")" "$EVAL_ONLY" || exit 1
    echo "${CYAN}verdict in $EVAL_ONLY/verdict.txt${RESET}"
    exit 0
fi

if (( $# > 0 )); then
    MODELS=("$@")
else
    [[ -f "$CONFIG" ]] || { echo "error: config not found: $CONFIG" >&2; exit 1; }
    command -v yq >/dev/null 2>&1 \
        || { echo "error: yq (mikefarah v4) is required to read $CONFIG" >&2; exit 1; }
    mapfile -t MODELS < <(models_from_config)
fi
if (( ${#MODELS[@]} == 0 )); then
    echo "error: no models found (pass aliases as arguments, or check $CONFIG)" >&2
    exit 1
fi

echo "Testing model(s):"
for model in "${MODELS[@]}"; do
    echo "    $model"
done
echo "Instance filter: $FILTER"

mkdir -p "$RESULTS_DIR"
declare -a FAILED=()
START_ALL=$SECONDS

for model in "${MODELS[@]}"; do
    echo "${CYAN}Testing model $model...${RESET}"
    warmup "$model"
    OUTDIR="$RESULTS_DIR/$model-$STAMP"
    LOG="$OUTDIR/console.log"
    mkdir -p "$OUTDIR"
    # litellm sends the part after openai/ as the request model, which is how
    # llama-swap routes to the right backend
    sed -e "s#__MODEL__#openai/$model#" -e "s#__API_BASE__#$BASE_URL#" \
        "$CONFIG_TPL" > "$OUTDIR/agent-config.yaml"
    RUN_START=$SECONDS
    echo "${CYAN}  output in $OUTDIR, logging to $LOG${RESET}"
    # pipefail makes the pipeline status mini-extra's, not tee's; --workers 1
    # because the servers run --parallel 1
    AGENT_OK=1
    if mini-extra swebench --config "$OUTDIR/agent-config.yaml" \
        --output "$OUTDIR" --subset multilingual --split test \
        --filter "$FILTER" --workers 1 2>&1 | tee "$LOG"; then
        echo "${CYAN}  agent done in $((SECONDS - RUN_START))s${RESET}"
    else
        echo "warning: model $model agent run FAILED, log: $LOG" >&2
        FAILED+=("$model (agent)")
        AGENT_OK=0
    fi
    # Eval step: grade preds.json with the repo's own tests in Docker.
    # The agent's "Submitted" status only means a patch was produced; this is
    # what answers pass/fail per instance.
    if [[ "${SKIP_EVAL:-0}" == 1 ]]; then
        echo "${CYAN}  SKIP_EVAL=1, skipping eval step${RESET}"
    elif (( AGENT_OK == 0 )) || [[ ! -f "$OUTDIR/preds.json" ]]; then
        echo "warning: model $model has no preds.json, skipping eval" >&2
        FAILED+=("$model (no preds)")
    else
        EVAL_START=$SECONDS
        run_eval "$model" "$OUTDIR" || true
        echo "${CYAN}  eval done in $((SECONDS - EVAL_START))s${RESET}"
    fi
    echo "${CYAN}Model $model done!${RESET}"
done

echo
echo "${CYAN}=== agentic summary: $((SECONDS - START_ALL))s total ===${RESET}"
if (( ${#FAILED[@]} > 0 )); then
    echo "failed runs:"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi
echo "all models passed (results in $RESULTS_DIR)"
