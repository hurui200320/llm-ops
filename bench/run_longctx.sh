#!/usr/bin/env bash
# Longctx speed runner (bench step 1): measures near-full-context generation
# speed (the compaction workload) for models deployed on llama-swap, wrapping
# speed/run_longctx.py. The Python is stdlib-only — nothing to install, and no
# bash-side warmup: its /props probe plus its own warmup request load the model.
#
# Usage:
#   ./run_longctx.sh                      # all models from deploy/llama-swap.config.yaml
#   ./run_longctx.sh <alias> [alias...]   # only the named llama-swap models
#
# Each run is teed to results/longctx/<model>-<stamp>.log with the per-rep
# results JSON next to it (both gitignored); the JSON is rewritten after every
# rep, so an interrupted run keeps its completed reps. Expect tens of minutes
# per rep: a cold ~full-context prefill plus a long decode.
#
# Env:
#   LLAMA_SWAP_URL  llama-swap root URL (default http://10.233.1.16:8080)
#   REPS            reps per model (default 2; each rep uses a different
#                   corpus segment, so reps are comparable but cold)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/../deploy/llama-swap.config.yaml"
RESULTS_DIR="$SCRIPT_DIR/results/longctx"
PY="$SCRIPT_DIR/speed/run_longctx.py"

LLAMA_SWAP_URL="${LLAMA_SWAP_URL:-http://10.233.1.16:8080}"
REPS="${REPS:-2}"
STAMP="$(date +%Y%m%d-%H%M%S)"

CYAN=$'\033[96m'
RESET=$'\033[0m'

models_from_config() {
    yq '.models | keys | .[]' "$CONFIG"
}

[[ -f "$PY" ]] || { echo "error: script not found: $PY" >&2; exit 1; }

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
declare -a FAILED=()
START_ALL=$SECONDS

for model in "${MODELS[@]}"; do
    echo "${CYAN}Testing model $model...${RESET}"
    JSON="$RESULTS_DIR/$model-$STAMP.json"
    LOG="$RESULTS_DIR/$model-$STAMP.log"
    RUN_START=$SECONDS
    echo "${CYAN}  reps=$REPS, results in $JSON, logging to $LOG${RESET}"
    # pipefail makes the pipeline status the python script's, not tee's; the
    # JSON is still written when a rep fails, only the exit status is non-zero
    if python3 "$PY" --model "$model" --base-url "$LLAMA_SWAP_URL" --reps "$REPS" \
        --out "$JSON" 2>&1 | tee "$LOG"; then
        echo "${CYAN}  done in $((SECONDS - RUN_START))s${RESET}"
    else
        echo "warning: model $model FAILED, log: $LOG" >&2
        FAILED+=("$model")
    fi
    echo "${CYAN}Model $model done!${RESET}"
done

echo
echo "${CYAN}=== longctx summary: $((SECONDS - START_ALL))s total ===${RESET}"
if (( ${#FAILED[@]} > 0 )); then
    echo "failed runs:"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi
echo "all models passed (results in $RESULTS_DIR)"
