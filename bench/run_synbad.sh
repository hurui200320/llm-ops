#!/usr/bin/env bash
# Synbad gate runner (bench step 0): detects serving-stack bugs (tool-call
# parsing, parallel tool calls, reasoning-field parsing, streaming) for
# models deployed on llama-swap. synbad itself is installed/updated on the
# fly via npm — never vendored. Per synbad's own rule every model is
# evaluated both with and without --stream; a model only passes if both
# runs pass.
#
# Usage:
#   ./run_synbad.sh                      # all models from deploy/llama-swap.config.yaml
#   ./run_synbad.sh <alias> [alias...]   # only the named llama-swap models
#
# Output is teed per run to results/synbad/<model>-<plain|stream>-<stamp>.log
# (gitignored) so flaky failures can be diffed later.
#
# Env:
#   LLAMA_SWAP_URL  llama-swap root URL, /v1 is appended (default http://10.233.1.16:8080)
#   COUNT           evals per model+mode (default 20; synbad's README says 40
#                   per mode is enough to surface the 5%-rate response-in-reasoning
#                   bug — re-run with COUNT=40 if a failure looks borderline)
#   LLAMA_SWAP_KEY  value sent as the API key (default dummy; llama-swap runs
#                   with apiKeys: [], but synbad insists on an env-var name)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/../deploy/llama-swap.config.yaml"
RESULTS_DIR="$SCRIPT_DIR/results/synbad"

LLAMA_SWAP_URL="${LLAMA_SWAP_URL:-http://10.233.1.16:8080}"
COUNT="${COUNT:-20}"
BASE_URL="$LLAMA_SWAP_URL/v1"
STAMP="$(date +%Y%m%d-%H%M%S)"

CYAN=$'\033[96m'
RESET=$'\033[0m'

ensure_synbad() {
    local npm_log
    if npm_log="$(npm install -g @syntheticlab/synbad@latest 2>&1)"; then
        echo "synbad installed/updated at $(command -v synbad)"
        return
    fi
    if command -v synbad >/dev/null 2>&1; then
        echo "warning: npm install failed, using existing synbad at $(command -v synbad)" >&2
        echo "$npm_log" >&2
    else
        echo "error: npm install -g @syntheticlab/synbad@latest failed and synbad is not on PATH" >&2
        echo "$npm_log" >&2
        exit 1
    fi
}

models_from_config() {
    # model aliases are the two-space-indented quoted keys under the top-level
    # "models:" map; the macro keys earlier in the file end in ": >" and the
    # guard exits at the first non-indented line after "models:"
    awk '
        /^models:/            { in_models = 1; next }
        in_models && /^[^ #]/ { exit }
        in_models && /^  "[^"]+":$/ {
            line = $0
            sub(/^  "/, "", line)
            sub(/":$/, "", line)
            print line
        }
    ' "$CONFIG"
}

warmup() {
    # llama-swap loads a model on its first request; trigger the load here so
    # a slow cold start can't be misread as a synbad failure
    local model="$1"
    curl -sf --max-time 600 "$BASE_URL/chat/completions" \
        -H "Authorization: Bearer $LLAMA_SWAP_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
        >/dev/null || echo "warning: warmup request for $model failed, continuing anyway" >&2
}

ensure_synbad
export LLAMA_SWAP_KEY="${LLAMA_SWAP_KEY:-dummy}"

if (( $# > 0 )); then
    MODELS=("$@")
else
    [[ -f "$CONFIG" ]] || { echo "error: config not found: $CONFIG" >&2; exit 1; }
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
declare -a FAILED=()
START_ALL=$SECONDS

for model in "${MODELS[@]}"; do
    echo "${CYAN}Testing model $model...${RESET}"
    warmup "$model"
    for mode in plain stream; do
        LOG="$RESULTS_DIR/$model-$mode-$STAMP.log"
        STREAM_ARGS=()
        if [[ $mode == stream ]]; then
            STREAM_ARGS=(--stream)
        fi
        RUN_START=$SECONDS
        echo "${CYAN}  $mode (count=$COUNT), logging to $LOG${RESET}"
        # pipefail makes the pipeline status synbad's, not tee's
        if synbad eval --env-var LLAMA_SWAP_KEY --base-url "$BASE_URL" --model "$model" \
            --count "$COUNT" "${STREAM_ARGS[@]}" 2>&1 | tee "$LOG"; then
            echo "${CYAN}  $mode done in $((SECONDS - RUN_START))s${RESET}"
        else
            echo "warning: model $model ($mode) FAILED, log: $LOG" >&2
            FAILED+=("$model ($mode)")
        fi
    done
    echo "${CYAN}Model $model done!${RESET}"
done

echo
echo "${CYAN}=== synbad summary: $((SECONDS - START_ALL))s total ===${RESET}"
if (( ${#FAILED[@]} > 0 )); then
    echo "failed runs:"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi
echo "all models passed (logs in $RESULTS_DIR)"
