#!/usr/bin/env bash
# Tool-eval-bench runner (bench step 2): runs the 69-scenario tool-calling
# quality benchmark for models deployed on llama-swap. tool-eval-bench itself
# is installed via uv (uv tool install git+https://github.com/SeraphimSerapis/
# tool-eval-bench.git) — never vendored; this script only extracts the model
# aliases, cds into bench/ (its runs/ + data/ artifacts land relative to the
# CWD, which must be the gitignored dirs) and loops.
#
# Usage:
#   ./run_tooleval.sh                      # all models from deploy/llama-swap.config.yaml
#   ./run_tooleval.sh <alias> [alias...]   # only the named llama-swap models
#
# Each run is teed to results/tooleval/<model>-<stamp>.log (gitignored); the
# markdown report and SQLite record land in bench/runs/ + bench/data/. Also
# see `tool-eval-bench compare <runA> <runB>` afterwards.
#
# Env:
#   LLAMA_SWAP_URL  llama-swap root URL (default http://10.233.1.16:8080; no
#                   /v1 — the bench appends its own paths)
#   SEED            passed as --seed (default 42; pin it across compared models)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/../deploy/llama-swap.config.yaml"
RESULTS_DIR="$SCRIPT_DIR/results/tooleval"

LLAMA_SWAP_URL="${LLAMA_SWAP_URL:-http://10.233.1.16:8080}"
SEED="${SEED:-42}"
STAMP="$(date +%Y%m%d-%H%M%S)"

# llama-swap is LAN; a shell-wide proxy (e.g. ALL_PROXY=socks://) must never
# apply here, and httpx errors out on socks:// schemes before no_proxy even
# matters (it does not honor CIDR no_proxy entries either)
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

CYAN=$'\033[96m'
RESET=$'\033[0m'

command -v tool-eval-bench >/dev/null 2>&1 || {
    echo "error: tool-eval-bench not on PATH; install with:" >&2
    echo "  uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git" >&2
    exit 1
}

models_from_config() {
    yq '.models | keys | .[]' "$CONFIG"
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

# tool-eval-bench writes runs/ and data/ relative to the CWD, so run from
# bench/ to keep those artifacts in the gitignored dirs
cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"
declare -a FAILED=()
START_ALL=$SECONDS

for model in "${MODELS[@]}"; do
    echo "${CYAN}Testing model $model...${RESET}"
    LOG="$RESULTS_DIR/$model-$STAMP.log"
    RUN_START=$SECONDS
    echo "${CYAN}  seed=$SEED, logging to $LOG${RESET}"
    # pipefail makes the pipeline status the bench's, not tee's
    # Note about the arg: the probe uses 256 budget to probe `tool_required`, that's not enough for model to make too call
    # so tool-eval-bench will mislabel the endpoint as not enforcing tool_required
    if TOOL_EVAL_MODEL="$model" tool-eval-bench run --base-url "$LLAMA_SWAP_URL" \
        --seed "$SEED" 2>&1 | tee "$LOG"; then
        echo "${CYAN}  done in $((SECONDS - RUN_START))s${RESET}"
    else
        echo "warning: model $model FAILED, log: $LOG" >&2
        FAILED+=("$model")
    fi
    echo "${CYAN}Model $model done!${RESET}"
done

echo
echo "${CYAN}=== tool-eval-bench summary: $((SECONDS - START_ALL))s total ===${RESET}"
if (( ${#FAILED[@]} > 0 )); then
    echo "failed runs:"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi
echo "all models passed (logs in $RESULTS_DIR)"
