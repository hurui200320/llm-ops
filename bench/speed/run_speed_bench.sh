#!/usr/bin/env bash
# SPEED-Bench runner for baseline-vs-speculative comparisons through llama-swap.
# Fetches the speed-bench client from llama.cpp master on every call (deliberately
# not pinned; the cached copy under bench/.cache/ is only used as an offline
# fallback) and runs it with uv (deps: datasets, requests, tqdm).
#
# Usage:
#   ./run_speed_bench.sh fetch                        # refresh client from master
#   ./run_speed_bench.sh run <alias> <tag> [args...]  # run bench for llama-swap
#                                                     # model <alias>, save as
#                                                     # results/speed/speed-bench-<tag>.json
#                                                     # extra args go to speed_bench.py
#   ./run_speed_bench.sh compare <a.json> <b.json>    # diff two saved runs
#
# Typical MTP/DFlash A/B:
#   1. add two aliases for the same model in llama-swap.config.yaml, one with
#      --spec-type none, one with spec enabled
#   2. ./run_speed_bench.sh run mymodel-nospec nospec
#   3. ./run_speed_bench.sh run mymodel-spec spec
#   4. ./run_speed_bench.sh compare ../results/speed/speed-bench-nospec.json \
#                                   ../results/speed/speed-bench-spec.json
#
# NOTE — unload all models on llama-swap before every run (GET /unload, or
# the UI). Requests keep the server's normal prompt caching (the client's
# own default extra-inputs only pins temperature 0), so multi-turn samples
# reuse the conversation prefix like real chat traffic; a model kept loaded
# holds a stale prompt cache / ctx checkpoints, and a cache hit would fake
# a fast result.
#
# Env:
#   LLAMA_SWAP_URL   default http://fedora-tuf:8080
#   SPEED_BENCH_REF  llama.cpp git ref to fetch the client from (default master)
#   LIMIT            samples per category (default 10; set empty for full)
#   OSL              output tokens per request (default 1024)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$SCRIPT_DIR/../.cache/speed-bench"
RESULTS_DIR="$SCRIPT_DIR/../results/speed"

LLAMA_SWAP_URL="${LLAMA_SWAP_URL:-http://fedora-tuf:8080}"
SPEED_BENCH_REF="${SPEED_BENCH_REF:-master}"
LIMIT="${LIMIT-10}"
OSL="${OSL:-1024}"

BASE_RAW="https://raw.githubusercontent.com/ggml-org/llama.cpp/${SPEED_BENCH_REF}/tools/server/bench/speed-bench"

fetch() {
    mkdir -p "$CACHE_DIR"
    for f in speed_bench.py speed_bench_compare.py; do
        if curl -sfL "$BASE_RAW/$f" -o "$CACHE_DIR/$f.tmp"; then
            mv "$CACHE_DIR/$f.tmp" "$CACHE_DIR/$f"
        else
            rm -f "$CACHE_DIR/$f.tmp"
            if [[ -f "$CACHE_DIR/$f" ]]; then
                echo "warning: could not fetch $f from $SPEED_BENCH_REF, using cached copy" >&2
            else
                echo "error: could not fetch $f and no cached copy present" >&2
                exit 1
            fi
        fi
    done
}

uvrun() {
    uv run --quiet --with datasets --with requests --with tqdm python3 "$@"
}

CMD="${1:-}"
case "$CMD" in
    fetch)
        fetch
        echo "cached in $CACHE_DIR"
        ;;
    run)
        ALIAS="${2:?usage: $0 run <alias> <tag> [args...]}"
        TAG="${3:?usage: $0 run <alias> <tag> [args...]}"
        shift 3
        fetch
        mkdir -p "$RESULTS_DIR"
        OUT="$RESULTS_DIR/speed-bench-${TAG}.json"
        LIMIT_ARGS=()
        if [[ -n "$LIMIT" ]]; then
            LIMIT_ARGS=(--limit "$LIMIT")
        fi
        uvrun "$CACHE_DIR/speed_bench.py" \
            --url "$LLAMA_SWAP_URL" \
            --model "$ALIAS" \
            --bench qualitative \
            --category all \
            --osl "$OSL" \
            --concurrency 1 \
            "${LIMIT_ARGS[@]}" \
            --output "$OUT" \
            "$@"
        echo "saved: $OUT"
        ;;
    compare)
        A="${2:?usage: $0 compare <baseline.json> <spec.json>}"
        B="${3:?usage: $0 compare <baseline.json> <spec.json>}"
        fetch
        uvrun "$CACHE_DIR/speed_bench_compare.py" --baseline "$A" --speculative "$B"
        ;;
    *)
        echo "usage: $0 fetch | run <alias> <tag> [args...] | compare <a.json> <b.json>" >&2
        exit 2
        ;;
esac
