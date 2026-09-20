#!/usr/bin/env bash
# Reasoning runner: runs the reasoning suite (12 light sanity problems from
# reasoning/problems.jsonl plus the frontier tier — AIME 2026 + generated
# zebra puzzles — from the gitignored cache built by reasoning/fetch_frontier.py)
# against models deployed on llama-swap, wrapping reasoning/run_reasoning.py.
# The Python is stdlib-only — nothing to install, and a bash-side warmup
# request loads the model so a cold start can't skew the first problem's
# timing.
#
# Usage:
#   ./run_reasoning.sh                      # all models from deploy/llama-swap.config.yaml
#   ./run_reasoning.sh <alias> [alias...]   # only the named llama-swap models
#
# Each run is teed to results/reasoning/<model>-<stamp>.log with the summary
# JSON next to it (both gitignored). Expect a while per model: problems are
# non-streaming and can run up to max-tokens each.
#
# Env:
#   LLAMA_SWAP_URL  llama-swap root URL (default http://10.233.1.16:8080; the
#                   /v1 suffix the Python expects is appended here)
#   MAX_TOKENS      passed as --max-tokens (default 131072)
#   SKIP_IDS        comma-separated problem ids excluded from runs. Default
#                   trims the comparison set to 35 of the 42 on disk (4
#                   redundant/saturated light problems + the 3 fastest clean
#                   AIME passes of the 31B calibration run — least information
#                   per second). The frozen manifest and the problem files
#                   still hold all 42, and the calibration run covers all of
#                   them, so comparisons stay paired on the shared subset. Set
#                   SKIP_IDS= (empty) to run everything.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/../deploy/llama-swap.config.yaml"
RESULTS_DIR="$SCRIPT_DIR/results/reasoning"
PY="$SCRIPT_DIR/reasoning/run_reasoning.py"
LIGHT="$SCRIPT_DIR/reasoning/problems.jsonl"
FRONTIER="$SCRIPT_DIR/.cache/reasoning/frontier.jsonl"

LLAMA_SWAP_URL="${LLAMA_SWAP_URL:-http://10.233.1.16:8080}"
MAX_TOKENS="${MAX_TOKENS:-131072}"
SKIP_IDS="${SKIP_IDS:-l01,l08,l09,l10,aime26-01,aime26-03,aime26-05}"
BASE_URL="$LLAMA_SWAP_URL/v1"
STAMP="$(date +%Y%m%d-%H%M%S)"

# llama-swap is LAN; a shell-wide proxy (e.g. ALL_PROXY=socks://) must never
# apply here, and urllib honors *_proxy env vars but cannot handle socks://
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

CYAN=$'\033[96m'
RESET=$'\033[0m'

[[ -f "$PY" ]] || { echo "error: script not found: $PY" >&2; exit 1; }

models_from_config() {
    yq '.models | keys | .[]' "$CONFIG"
}

warmup() {
    # llama-swap loads a model on its first request; trigger the load here so
    # a slow cold start can't be misread as slow first-problem timing
    local model="$1"
    curl -sf --max-time 600 "$BASE_URL/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
        >/dev/null || echo "warning: warmup request for $model failed, continuing anyway" >&2
}

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

mkdir -p "$RESULTS_DIR"
PROBLEM_ARGS=("$LIGHT")
if [[ -f "$FRONTIER" ]]; then
    PROBLEM_ARGS+=("$FRONTIER")
    # a frontier set that drifted from the frozen manifest would make results
    # incomparable with runs pinned to it — refuse to run silently on that
    if check_out="$(python3 "$SCRIPT_DIR/reasoning/fetch_frontier.py" --check 2>&1)"; then
        echo "frontier set verified against frontier_manifest.json"
    else
        echo "error: $check_out" >&2
        echo "       refetch (python3 reasoning/fetch_frontier.py) and re-freeze before comparing" >&2
        exit 1
    fi
else
    echo "warning: $FRONTIER missing — running the light sanity tier only." >&2
    echo "         build it first: python3 reasoning/fetch_frontier.py" >&2
fi
declare -a FAILED=()
SKIP_ARGS=()
[[ -n "$SKIP_IDS" ]] && SKIP_ARGS=(--skip "$SKIP_IDS")
START_ALL=$SECONDS

for model in "${MODELS[@]}"; do
    echo "${CYAN}Testing model $model...${RESET}"
    warmup "$model"
    JSON="$RESULTS_DIR/$model-$STAMP.json"
    LOG="$RESULTS_DIR/$model-$STAMP.log"
    RUN_START=$SECONDS
    echo "${CYAN}  max-tokens=$MAX_TOKENS, results in $JSON, logging to $LOG${RESET}"
    # pipefail makes the pipeline status the python script's, not tee's; the
    # summary JSON is still written when some requests fail, only the exit
    # status is non-zero
    if python3 "$PY" --base-url "$BASE_URL" --model "$model" \
        --problems "${PROBLEM_ARGS[@]}" "${SKIP_ARGS[@]}" \
        --max-tokens "$MAX_TOKENS" --out "$JSON" 2>&1 | tee "$LOG"; then
        echo "${CYAN}  done in $((SECONDS - RUN_START))s${RESET}"
    else
        echo "warning: model $model FAILED, log: $LOG" >&2
        FAILED+=("$model")
    fi
    echo "${CYAN}Model $model done!${RESET}"
done

echo
echo "${CYAN}=== reasoning summary: $((SECONDS - START_ALL))s total ===${RESET}"
if (( ${#FAILED[@]} > 0 )); then
    echo "failed runs:"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi
echo "all models passed (results in $RESULTS_DIR)"
