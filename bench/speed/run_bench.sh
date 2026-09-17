#!/usr/bin/env bash
# Run llama-bench for a model from models.conf, inside the same llama.cpp
# docker image used for deployment (tools.sh entrypoint maps --bench to
# llama-bench). Run this on the GPU host (fedora-tuf).
#
# Usage:
#   ./run_bench.sh <model-key>        # keys: ./run_bench.sh --list
#
# Env overrides:
#   MODELS_DIR   host model dir mounted to /models  (default /mnt/unenc-xfs/models)
#   RESULTS_DIR  where CSV goes                     (default bench/results/speed)
#   IMAGE        full image name override           (default ghcr.io/ggml-org/llama.cpp:full-<backend>)
#   PROMPT/GEN   -p / -n                            (default 512 / 128)
#   REPS         -r repetitions                     (default 3)
#
# Output: CSV written to RESULTS_DIR/<key>-<timestamp>.csv, md table to stderr.
# Note: the deepest point dominates runtime (up to 235K prefill per test x reps).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=models.conf
source "$SCRIPT_DIR/models.conf"

MODELS_DIR="${MODELS_DIR:-/mnt/unenc-xfs/models}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/../results/speed}"
PROMPT="${PROMPT:-512}"
GEN="${GEN:-128}"
REPS="${REPS:-3}"

if [[ "${1:-}" == "--list" || "${1:-}" == "-l" ]]; then
    model_keys
    exit 0
fi

KEY="${1:-}"
if [[ -z "$KEY" ]] || ! model_config "$KEY"; then
    echo "usage: $0 <model-key>   (or --list)" >&2
    exit 2
fi

IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:full-${BACKEND}}"

if [[ "$BACKEND" == "rocm" ]]; then
    DEVICES=(--device /dev/dri --device /dev/kfd --group-add=video --group-add=render)
elif [[ "$BACKEND" == "vulkan" ]]; then
    DEVICES=(--device /dev/dri --group-add=video --group-add=render)
else
    echo "unknown BACKEND: $BACKEND" >&2
    exit 2
fi

mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/${KEY}-$(date +%Y%m%d-%H%M%S).csv"

echo "model:   $MODEL"
echo "image:   $IMAGE"
echo "flags:   $FLAGS"
echo "depths:  $DEPTHS  (pp$PROMPT / tg$GEN, $REPS reps)"
echo "csv:     $OUT"

# FLAGS is intentionally word-split (it is a flag list).
# -o csv goes to the results file; -oe md prints a readable table to the terminal.
# shellcheck disable=SC2086
docker run --rm \
    "${DEVICES[@]}" \
    --security-opt label=disable \
    --security-opt seccomp=unconfined \
    -v "$MODELS_DIR:/models:z" \
    "$IMAGE" --bench \
    -m "$MODEL" \
    $FLAGS \
    -p "$PROMPT" -n "$GEN" -d "$DEPTHS" \
    -r "$REPS" \
    -o csv -oe md \
    > "$OUT"

echo "done: $OUT"
