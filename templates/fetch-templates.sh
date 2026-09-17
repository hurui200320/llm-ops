#!/bin/bash

# Fetch the latest chat templates from Hugging Face into this directory.
#
# Each entry below is a "relative/path|URL" pair: the path is relative to this
# script's directory, the URL must point at the file to download (resolve/main
# always serves the latest revision). Gated repos (google/gemma-*) need a
# token: export HF_TOKEN=... before running. Re-running re-syncs to latest.

set -euo pipefail

if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required but not found on PATH" >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)

entries=(
    "gemma-4/26B-chat-template.jinja|https://huggingface.co/google/gemma-4-26B-A4B-it/resolve/main/chat_template.jinja"
    "gemma-4/31B-chat-template.jinja|https://huggingface.co/google/gemma-4-31B-it/resolve/main/chat_template.jinja"
    "muse-glimmer/30B-chat-template.jinja|https://huggingface.co/meta-models/Muse-Glimmer-30B/resolve/main/chat_template.jinja"
    "ornith-1.5/35B-chat-template.jinja|https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/resolve/main/chat_template.jinja"
)

curl_args=(-fsSL --retry 3 --connect-timeout 15 --max-time 120)
if [[ -n "${HF_TOKEN:-}" ]]; then
    curl_args+=(-H "Authorization: Bearer $HF_TOKEN")
fi

failed=0
for entry in "${entries[@]}"; do
    if [[ "$entry" != *"|"* ]]; then
        echo "ERROR: malformed entry, expected 'relative/path|URL': $entry" >&2
        failed=$((failed + 1))
        continue
    fi
    dest="${entry%%|*}"
    url="${entry#*|}"
    if [[ -z "$dest" || "$dest" == /* || "$dest" == *".."* ]]; then
        echo "ERROR: invalid destination path in entry: '$dest'" >&2
        failed=$((failed + 1))
        continue
    fi
    target="$SCRIPT_DIR/$dest"
    tmp="$target.tmp"
    mkdir -p -- "$(dirname -- "$target")"
    echo "Fetching $dest"
    if curl "${curl_args[@]}" -o "$tmp" "$url"; then
        if [[ -s "$tmp" ]]; then
            mv -f "$tmp" "$target"
            echo "OK: $dest"
        else
            rm -f -- "$tmp"
            echo "ERROR: downloaded file is empty: $url" >&2
            failed=$((failed + 1))
        fi
    else
        status=$?
        rm -f -- "$tmp"
        echo "ERROR: curl exited $status for $url" >&2
        if [[ $status -eq 22 ]]; then
            echo "Hint: HTTP error (401/403/404). Gated repos need an HF token and accepted license: export HF_TOKEN=..." >&2
        fi
        failed=$((failed + 1))
    fi
done

if [[ $failed -gt 0 ]]; then
    echo "$failed of ${#entries[@]} template(s) failed" >&2
    exit 1
fi
echo "All ${#entries[@]} template(s) synced"
