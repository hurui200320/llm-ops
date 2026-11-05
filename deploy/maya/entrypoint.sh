#!/bin/sh
# Entrypoint for the llm-ops Maya image (deploy/maya/Dockerfile).
#
# No arguments: serve the newest installed config (maya-*-hip.json kept on /data/config).  Any arguments are
# passed to maya.py unchanged, which is how the setup is run - once with a model you already have:
#
#   docker run ... IMAGE --setup --backend hip --gpu 0 --yes --no-start \
#       --gguf-dir /data/models/GLM-5.3-Flash-Maya-S --host 0.0.0.0 --port 8080
#   docker run ... IMAGE --check --backend hip --gpu 0
#   docker run ... IMAGE --bench --backend hip
#
# Environment (no-argument start):
#   MODEL       setup id to serve (default glm-maya-s); only consulted when no config exists yet
#   SETUP=1     first run without a config: run the setup (add DOWNLOAD=1 to download the model, ~96 GB)
#   CONTEXT     context length for that setup, e.g. 262144
#   GPU / GPUS  one card (default 0) or a layer split, e.g. GPUS=0,1
#   PORT/HOST   where the dashboard and the API listen (default 8080 / 0.0.0.0)
#   API_KEY     require this key from API clients
#   MODELS_DIR  where the SETUP path installs models (default /data/models; nothing else reads it)
#   PACK_BUILD=1  the read-only model-mount layout (the DeepSeek lane): the GGUF shards stay in the production
#                 folder on the host, bind-mounted read-only at /data/model; the pack reads them in place
#                 (ds4.txt records model_dir /data/model) and the vision tower is looked up beside them as
#                 ../vision/vision.bin (a path relative to model_dir, resolved at every load). Everything
#                 derived - the tower in /data/vision, the pack in /data/pack/maya-ds4 - is built into the
#                 writable /data when missing: a fresh data tree self-provisions at the first start, and a
#                 deleted tower or pack is rebuilt at the next one. Without PACK_BUILD a missing pack or tower
#                 stops the start with the fix instead of the engine's load error.
#   DS4_VISION_TOKENS  DeepSeek picture budget (prompt tokens): written into every maya-ds4 pack's ds4.txt
#                      (vision.max_image_tokens) at every start. The tower's hard maximum is 1140 (10,240-patch
#                      attention); a larger value is clamped with a warning. Unset: the pack keeps its own value
#                      (pack_gguf.py writes Maya's default 1024). The strata fork's change 6, mirrored.
#   DS4_VISION_MIN_PIXELS  DeepSeek: upscale a small picture to at least this many pixels (the pack's
#                      vision.min_pixels). Unset: the pack keeps its own value.
#
# Everything the container writes lives in /data: configs in /data/config (maya.py writes them into its own
# folder, which is the container layer), the derived pack and vision tower beside the read-only shards at
# /data/model, SETUP-installed models in MODELS_DIR.
set -eu

cd /opt/maya

if [ "$#" -gt 0 ]; then
    exec /opt/maya/.venv/bin/python maya.py "$@"
fi

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
MODEL="${MODEL:-glm-maya-s}"
MODELS_DIR="${MODELS_DIR:-/data/models}"
CONFIG_DIR=/data/config
MODEL_MOUNT=/data/model         # the shards' read-only mount (PACK_BUILD layout)
VISION_DIR=/data/vision         # the tower: a sibling of the mount, ../vision from model_dir
PACK_DIR=/data/pack/maya-ds4    # the pack: ds4.txt, tensors.txt, engram.bin, tokenizer/

mkdir -p "$CONFIG_DIR"

# CONFIG_SRC: a folder of config JSONs to install into /data/config before serving - how a lane carries its
# engine settings in the repo (each llama-swap lane bind-mounts its own deploy/maya/configs/<lane> folder read-only
# here). Copied at every start: the repo file is the declared state (the strata lane's STRATA_SET equivalent), so
# dashboard edits last until the next lane restart. One maya lane per CONFIG_SRC folder (a plain copy refreshes every
# JSON's mtime, and the serve picks the newest).
# A live config that differs from the repo copy is summarized BEFORE it is overwritten: maya.py rewrites the
# config on schema migrations (an upstream bump) and the dashboard edits it - both are reset by this copy, and
# this line is the only way a maya-side change is noticed at all (port it into the repo file to keep it).
CONFIG_SRC="${CONFIG_SRC:-}"
if [ -n "$CONFIG_SRC" ] && [ -d "$CONFIG_SRC" ]; then
    installed=0
    for f in "$CONFIG_SRC"/*.json; do
        [ -f "$f" ] || continue
        installed=$((installed + 1))
        name=$(basename "$f")
        if [ -f "$CONFIG_DIR/$name" ] && ! cmp -s "$f" "$CONFIG_DIR/$name"; then
            summary="$(python3 - "$f" "$CONFIG_DIR/$name" <<'PY'
import json, sys

def load(p):
    try:
        with open(p, encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None

def brief(v):
    s = v if isinstance(v, str) else json.dumps(v)
    return s if len(s) <= 40 else s[:37] + "..."

repo, live = load(sys.argv[1]), load(sys.argv[2])
if repo is None or live is None:
    print("unreadable json")
    raise SystemExit(0)
parts = []
for k in sorted(set(repo) | set(live)):
    if k not in live:
        parts.append("only in repo: " + k)
    elif k not in repo:
        parts.append("only in live: " + k)
    elif repo[k] == live[k]:
        continue
    elif isinstance(repo[k], list) and isinstance(live[k], list) and \
            all(isinstance(x, str) for x in repo[k] + live[k]):
        d = []
        if [x for x in repo[k] if x not in live[k]]:
            d.append("-" + " -".join(brief(x) for x in repo[k] if x not in live[k]))
        if [x for x in live[k] if x not in repo[k]]:
            d.append("+" + " +".join(brief(x) for x in live[k] if x not in repo[k]))
        parts.append(k + " (" + "; ".join(d) + ")")
    else:
        parts.append(k + ": " + brief(live[k]))
print("; ".join(parts) if parts else "differs")
PY
)"
            echo "maya: /data/config/$name differs from the repo copy ($summary). The repo copy replaces it:" \
                "dashboard edits and maya's own rewrites are reset - port wanted changes into the repo file."
        fi
        cp "$f" "$CONFIG_DIR/$name"
    done
    if [ "$installed" -gt 0 ]; then
        echo "maya: installed $(basename "$CONFIG_SRC" 2>/dev/null || echo "$CONFIG_SRC")/*.json into $CONFIG_DIR"
    else
        echo "maya: CONFIG_SRC=$CONFIG_SRC holds no *.json - nothing installed (mount the folder that holds the configs directly)." >&2
    fi
fi

# ---- the read-only model mount (PACK_BUILD layout). The shards stay in the production folder on the host and are
# bind-mounted read-only at /data/model; the engine opens them in place through the pack's model_dir. Everything
# derived (the vision tower, the pack) is built into the writable data tree when missing, so a fresh tree
# self-provisions and a deleted tower or pack is rebuilt at the next start. The tower's folder is recorded in the
# pack as ../vision (relative to model_dir) - that is why the shards are mounted INSIDE the writable /data instead
# of a shared read-only /models mount like strata's: ../vision must resolve into a writable folder, and the tower
# itself (vision.bin, built from the mmproj) must not be written beside the production shards.
if [ -d "$MODEL_MOUNT" ]; then
    if ! ls "$MODEL_MOUNT"/*.gguf >/dev/null 2>&1; then
        echo "maya: $MODEL_MOUNT holds no .gguf files. Mount the model's shard folder read-only there, e.g." \
            "-v /mnt/unenc-xfs/models/maya/<model>:$MODEL_MOUNT:ro. Nothing was started." >&2
        exit 2
    fi

    build_vision() {
        mmproj="$(ls "$MODEL_MOUNT"/mmproj*.gguf 2>/dev/null | head -n 1 || true)"
        if [ -z "$mmproj" ]; then
            echo "maya: $MODEL_MOUNT has no mmproj*.gguf to build the vision tower from. Nothing was started." >&2
            exit 2
        fi
        mkdir -p "$VISION_DIR"
        echo "maya: building the vision tower into $VISION_DIR (from ${mmproj#$MODEL_MOUNT/})"
        /opt/maya/.venv/bin/python /opt/maya/tools/ds4/vision_pack.py --dir "$VISION_DIR" --mmproj "$mmproj"
    }

    if [ ! -f "$PACK_DIR/ds4.txt" ]; then
        if [ "${PACK_BUILD:-0}" != "1" ]; then
            echo "maya: no pack at $PACK_DIR yet. Set PACK_BUILD=1 to build the vision tower and the pack from" \
                "the shards in $MODEL_MOUNT (a fresh data tree self-provisions this way). Nothing was started." >&2
            exit 2
        fi
        [ -f "$VISION_DIR/vision.bin" ] || build_vision
        # The shards are named explicitly, because the model folder also holds the vision tower's source
        # (mmproj-*.gguf, architecture clip) and pack_gguf.py would take every .gguf in the folder as a shard
        # otherwise. Everything else belongs to the split - including ISTA's sixth shard, which is published
        # under the Engram name (split.no 5) - and a true side Engram set GGUF is pack_gguf.py's own to filter.
        shard_list=""
        for g in "$MODEL_MOUNT"/*.gguf; do
            [ -f "$g" ] || continue
            case "${g#$MODEL_MOUNT/}" in mmproj*) continue ;; esac
            shard_list="$shard_list ${g#$MODEL_MOUNT/}"
        done
        if [ -z "$shard_list" ]; then
            echo "maya: $MODEL_MOUNT holds no shards (only an mmproj?) - name them for pack_gguf.py by hand" \
                "(tools/ds4/pack_gguf.py --shards). Nothing was started." >&2
            exit 2
        fi
        echo "maya: building the pack into $PACK_DIR (reads the shards' headers only)"
        # shellcheck disable=SC2086  # the shard names are single words by construction (no spaces in the names)
        /opt/maya/.venv/bin/python /opt/maya/tools/ds4/pack_gguf.py --model "$MODEL_MOUNT" --out "$PACK_DIR" \
            --vision "$VISION_DIR" --shards$shard_list
    fi

    # The pack names the shards' folder (model_dir in ds4.txt): point every maya-ds4 pack under /data at the
    # mount, so a pack built elsewhere (a host-side rebuild) still resolves inside the container. A pack with a
    # vision tower whose tower file is missing exits 3: rebuilt below when PACK_BUILD=1, else the start stops
    # here (the engine would fail at the load with less to go on).
    rc=0
    norm="$(python3 - /data "$MODEL_MOUNT" <<'PY'
import os, pathlib, sys

root, want = pathlib.Path(sys.argv[1]), sys.argv[2]
msgs, no_vision = [], []
for p in sorted(root.glob("**/maya-ds4/ds4.txt")):
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    keys, out, rewrote = {}, [], False
    for ln in lines:
        k, _, v = ln.partition(" ")
        if k == "model_dir" and v.strip() != want:
            out.append("model_dir " + want)
            rewrote = True
        else:
            out.append(ln)
        if k:
            keys[k] = v.strip()
    if rewrote:
        tmp = str(p) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        os.replace(tmp, p)
        msgs.append(str(p.relative_to(root)) + ": model_dir -> " + want)
    try:
        tower = int(keys.get("vision.num_hidden_layers", "0") or 0) > 0 and "image_token_id" in keys
    except ValueError:
        tower = False
    if tower and not os.path.isfile((want if rewrote else keys.get("model_dir", want)) + "/../vision/vision.bin"):
        no_vision.append(str(p))
if msgs:
    print("\n".join(msgs))
if no_vision:
    sys.stderr.write("maya: the vision tower is missing for: " + ", ".join(no_vision) + "\n")
raise SystemExit(3 if no_vision else 0)
PY
)" || rc=$?
    if [ -n "$norm" ]; then
        echo "maya: $norm"
    fi
    if [ "$rc" -eq 3 ]; then
        if [ "${PACK_BUILD:-0}" = "1" ]; then
            echo "maya: the pack's vision tower is missing under $VISION_DIR (PACK_BUILD=1: rebuilding it)."
            build_vision
        else
            echo "maya: the pack's vision tower is missing under $VISION_DIR. Set PACK_BUILD=1 to rebuild it" \
                "from the model mount's mmproj. Nothing was started." >&2
            exit 2
        fi
    elif [ "$rc" -ne 0 ]; then
        exit "$rc"
    fi
fi

gpus="--gpu ${GPU:-0}"
[ -n "${GPUS:-}" ] && gpus="--gpus ${GPUS}"
context=""
[ -n "${CONTEXT:-}" ] && context="--context ${CONTEXT}"
key=""
[ -n "${API_KEY:-}" ] && key="--api-key ${API_KEY}"

# the config maya.py wrote last (it touches the file on every start): the most recently used model
cfg="$(ls -1t "$CONFIG_DIR"/maya-*-hip.json 2>/dev/null | head -n 1 || true)"

if [ -z "$cfg" ]; then
    if [ "${SETUP:-0}" != "1" ]; then
        cat >&2 <<'EOF'
maya: no config in /data/config yet - run the setup once, then this container serves it.
  with a model you already have (the pack is written into that folder, it must be writable):
    docker run ... IMAGE --setup --backend hip --gpu 0 --yes --no-start \
        --gguf-dir /data/models/GLM-5.3-Flash-Maya-S --host 0.0.0.0 --port 8080
  or let the container download it (SETUP=1 DOWNLOAD=1, a Maya quant id in MODEL):
    docker run ... -e SETUP=1 -e DOWNLOAD=1 -e MODEL=glm-maya-s IMAGE
EOF
        exit 2
    fi
    echo "maya: first start - setting up ${MODEL} (the engine is prebuilt; nothing is compiled)"
    download=""
    [ "${DOWNLOAD:-0}" = "1" ] && download="--download-model"
    # shellcheck disable=SC2086  # the flags above are single words by construction
    /opt/maya/.venv/bin/python maya.py --setup --backend hip $gpus $context $key \
        --yes --model "$MODEL" --models-dir "$MODELS_DIR" $download \
        --no-start --host "$HOST" --port "$PORT"
    cp -p /opt/maya/maya-*-hip.json "$CONFIG_DIR"/ 2>/dev/null || true
    cfg="$(ls -1t "$CONFIG_DIR"/maya-*-hip.json 2>/dev/null | head -n 1 || true)"
    if [ -z "$cfg" ]; then
        echo "maya: the setup finished without a config (see above) - the model files were not found or downloaded" >&2
        exit 2
    fi
fi

# A config that serves a pack from /data/pack needs the shard mount: the engine opens the GGUFs through the
# pack's model_dir (/data/model). Fail with the fix instead of the engine's load error.
if [ -n "$cfg" ] && [ ! -d "$MODEL_MOUNT" ]; then
    wants_mount="$(python3 - "$cfg" <<'PY'
import json, sys
args = (json.load(open(sys.argv[1], encoding="utf-8-sig")).get("args") or [])
print("1" if any(a == "--pack" and str(p).startswith("/data/pack/") for a, p in zip(args, args[1:])) else "")
PY
)"
    if [ -n "$wants_mount" ]; then
        echo "maya: $(basename "$cfg") serves a pack from /data/pack, but $MODEL_MOUNT is not mounted - add" \
            "-v <shards folder>:$MODEL_MOUNT:ro (deploy/maya/README.md). Nothing was started." >&2
        exit 2
    fi
fi

# ---- DeepSeek picture budget (the strata fork's change 6, mirrored). The engine carries its own vision tower and
# sizes each picture from the pack's ds4.txt (vision.max_image_tokens; one prompt token per 42x42 px cell, span
# n_h*(n_w+1)+2) - there is no engine flag or config key for it, the pack file is the only knob. pack_gguf.py writes
# Maya's default 1024, which downscales a 4K screenshot to 1680x966; the tower's hard maximum is 1140 (10,240 patches
# in the attention softmax, kSmT*kSmPer in src/ds4/ds4_vision.cu; a larger value stops the tower loading with
# "more than the attention covers"), so a larger value is clamped. Writing at every start keeps the repo (the
# llama-swap lane's -e) the source of truth and survives a repack: maya.py re-runs pack_gguf.py when the pack tools'
# source hash changes (an upstream bump), which would reset the key to 1024 - the next start re-applies the budget.
if [ -n "${DS4_VISION_TOKENS:-}" ] || [ -n "${DS4_VISION_MIN_PIXELS:-}" ]; then
    for vname in DS4_VISION_TOKENS DS4_VISION_MIN_PIXELS; do
        eval "v=\${$vname:-}"
        [ -n "$v" ] || continue
        case "$v" in *[!0-9]*) echo "maya: $vname=$v is not a whole number (0 = no limit). Nothing was started." >&2; exit 2 ;; esac
    done
    if [ -n "${DS4_VISION_TOKENS:-}" ] && [ "$DS4_VISION_TOKENS" -gt 1140 ]; then
        echo "maya: DS4_VISION_TOKENS=$DS4_VISION_TOKENS is above the DeepSeek tower's 10,240-patch attention limit (max_image_tokens 1140): using 1140." >&2
        DS4_VISION_TOKENS=1140
    fi
    if ! changed="$(python3 - /data "${DS4_VISION_TOKENS:-}" "${DS4_VISION_MIN_PIXELS:-}" <<'PY'
import os, pathlib, sys

root = pathlib.Path(sys.argv[1])
want = {}
if sys.argv[2]:
    want["vision.max_image_tokens"] = sys.argv[2]
if sys.argv[3]:
    want["vision.min_pixels"] = sys.argv[3]
for p in sorted(root.glob("**/maya-ds4/ds4.txt")):
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    keys = {}
    for ln in lines:
        k, _, v = ln.partition(" ")
        if k:
            keys[k] = v.strip()
    try:
        tower = int(keys.get("vision.num_hidden_layers", "0") or 0) > 0 and "image_token_id" in keys
    except ValueError:
        tower = False
    if not tower:
        continue                                  # not a DeepSeek pack with a vision tower
    todo = {k: v for k, v in want.items() if keys.get(k) != v}
    if not todo:
        continue                                  # already the wanted budget: leave the file untouched
    seen = set()
    out = []
    for ln in lines:
        k = ln.partition(" ")[0]
        if k in todo:
            out.append(k + " " + todo[k])
            seen.add(k)
        else:
            out.append(ln)
    for k, v in todo.items():
        if k not in seen:
            out.append(k + " " + v)
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    os.replace(tmp, p)
    print(str(p.relative_to(root)) + ": " + ", ".join(k + "=" + v for k, v in todo.items()))
PY
    )"; then
        echo "maya: could not write the DeepSeek picture budget into the pack under /data. Nothing was started." >&2
        exit 2
    fi
    if [ -n "$changed" ]; then
        echo "maya: DeepSeek vision budget: $changed"
    elif ! find /data -maxdepth 5 -type f -path '*/maya-ds4/ds4.txt' -print -quit 2>/dev/null | grep -q .; then
        echo "maya: DS4_VISION_TOKENS/DS4_VISION_MIN_PIXELS is set, but there is no maya-ds4 pack under /data yet: setup writes its default (1024) this start; the budget is applied on the next start." >&2
    fi
fi

echo "maya: serving $(basename "$cfg") on ${HOST}:${PORT} (data in /data)"
# shellcheck disable=SC2086
exec /opt/maya/.venv/bin/python maya.py --backend hip --config "$cfg" --host "$HOST" --port "$PORT" $key
