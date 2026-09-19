#!/usr/bin/env python3
"""Measure near-full-context speed through llama-swap (the compaction workload).

Defaults target llama-swap on the LAN. Standard library only.

Usage:
    python3 run_longctx.py --model <llama-swap-alias> [--ctx N]
                           [--isl N --osl N] [--reps 2]
                           [--base-url http://fedora-tuf:8080]
                           [--corpus <path-or-url>] [--out results.json]

Sends one chat completion per rep: a target of 85% of the max context of
meaningful text (a public-domain novel, fetched on the fly and cached under
bench/.cache/) wrapped in a summarize instruction — the shape of a harness
compaction request — with a 10% of max context output budget. Reps use
different corpus segments and cache_prompt=false, so every prefill is cold
(no KV reuse / checkpoint-restore skew). Draft acceptance only means
something on meaningful input, which is why real text is used instead of
synthetic filler; run with MTP enabled first and watch VRAM (see
bench/README.md). Per-rep prefill/decode t/s, latency, speculative draft
acceptance and the cached-prefix token count (cache_n — 0 proves the prefill
was cold) come from llama-server response timings; actual token counts come
from response usage. The max context is auto-detected from llama-server
/props through llama-swap's /upstream/<model> passthrough
(default_generation_settings.n_ctx — per-slot when parallel slots are
configured, which is the right cap for a single request; llama-swap does not
route plain /props); --ctx overrides it with a warning when the two disagree
and is required for backends that do not serve /props. Prompt sizing uses
llama-server /tokenize through the same passthrough (llama-swap does not
route plain /tokenize; a bare llama-server is covered by the plain path, and
the warmup request makes sure the right model is loaded either way), with a
max-tokens-1 calibration request as last resort. The /props probe records
the loaded model's n_ctx, slot count, quant and model path into the results
config for provenance. Results land in
bench/results/longctx/ and are rewritten after every rep, so an
interrupted run keeps its completed reps.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "longctx"
CACHE_PATH = SCRIPT_DIR.parent / ".cache" / "longctx" / "war_and_peace.txt"

CORPUS_URLS = [
    "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
    "https://www.gutenberg.org/files/2600/2600-0.txt",
]
USER_AGENT = "llm-ops-bench/1.0"
CHUNK_CHARS = 4000
CHARS_PER_TOKEN = 4.0
WARMUP_TIMEOUT = 600
TOKENIZE_TIMEOUT = 120

INSTRUCTION = (
    "You are given the full text of a very long document. Write an exhaustive, "
    "detailed summary of it: go through it from beginning to end and cover every "
    "major event, character and development, in order. Do not skip any part.\n\n"
    "=== DOCUMENT START ===\n\n"
)
INSTRUCTION_SUFFIX = "\n\n=== DOCUMENT END ==="


def post_json(url, payload, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get_json(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def props_fields(props):
    fields = {}
    if not isinstance(props, dict):
        return fields
    gen = props.get("default_generation_settings")
    n_ctx = gen.get("n_ctx") if isinstance(gen, dict) else None
    if not isinstance(n_ctx, int) or n_ctx <= 0:
        n_ctx = props.get("n_ctx")
    if isinstance(n_ctx, int) and n_ctx > 0:
        fields["n_ctx"] = n_ctx
    for key in ("total_slots", "model_alias", "model_ftype", "model_path"):
        if props.get(key) is not None:
            fields[key] = props[key]
    return fields


def http_error_text(exc):
    try:
        raw = exc.read().decode("utf-8", "replace").strip()
        if raw:
            body = raw[:800].replace("\n", " ").strip()
            return body + (" …" if len(raw) > 800 else "")
    except Exception:
        pass
    return str(exc.reason)


def split_base_url(url):
    url = url.strip().rstrip("/")
    if "://" not in url:
        url = "http://" + url
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"invalid --base-url: {url}")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def fetch_url(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace").replace("\r\n", "\n")


def load_corpus(source):
    if source:
        path = Path(source)
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
        if source.startswith(("http://", "https://")):
            return fetch_url(source, 120)
        raise SystemExit(f"error: --corpus {source!r} is neither a file nor a URL")
    if CACHE_PATH.is_file():
        return CACHE_PATH.read_text(encoding="utf-8", errors="replace")
    last_error = None
    for url in CORPUS_URLS:
        try:
            text = fetch_url(url, 120)
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:
            last_error = exc
    raise SystemExit(f"error: corpus fetch failed ({last_error}); no cached copy at {CACHE_PATH}")


def strip_gutenberg(text):
    start = text.find("*** START OF")
    if start != -1:
        newline = text.find("\n", start)
        text = text[newline + 1:] if newline != -1 else ""
    end = text.find("*** END OF")
    if end != -1:
        text = text[:end]
    return text


def chunk_text(text, chunk_chars):
    chunks = []
    pos, n = 0, len(text)
    while pos < n:
        end = min(pos + chunk_chars, n)
        if end < n:
            lo = pos + chunk_chars // 2
            cut = max(text.rfind(" ", lo, end), text.rfind("\n", lo, end))
            if cut > lo:
                end = cut
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)
        pos = end
    return chunks


def build_content(chunks):
    return INSTRUCTION + "\n\n".join(chunks) + INSTRUCTION_SUFFIX


def count_tokens(tokenize_urls, content):
    for url in tokenize_urls:
        try:
            data = post_json(url, {"content": content}, TOKENIZE_TIMEOUT)
            tokens = data.get("tokens") if isinstance(data, dict) else None
            if isinstance(tokens, list):
                return len(tokens)
        except Exception:
            pass
    return None


def warmup(chat_url, model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
        "stream": False,
    }
    started = time.perf_counter()
    post_json(chat_url, payload, WARMUP_TIMEOUT)
    print(f"warmup ok in {time.perf_counter() - started:.1f}s (model ready via llama-swap)")


def calibrate_chunks(chunks, isl, tokenize_urls, chat_url, model, args):
    total_chars = sum(len(c) for c in chunks)
    n = max(1, round(isl / (total_chars / len(chunks) / CHARS_PER_TOKEN)))
    count = count_tokens(tokenize_urls, build_content(chunks[:n]))
    if count is not None:
        for _ in range(2):
            if abs(count - isl) <= max(64, isl // 100):
                break
            new_n = max(1, round(n * isl / count))
            if new_n == n:
                break
            new_count = count_tokens(tokenize_urls, build_content(chunks[:new_n]))
            if new_count is None:
                print("warning: /tokenize failed mid-calibration; "
                      "keeping the last verified size", file=sys.stderr)
                break
            n, count = new_n, new_count
        print(f"sized: {n} chunks ~ {count} tokens via /tokenize")
        return n
    print("warning: /tokenize unavailable on this server; sizing via calibration request", file=sys.stderr)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": build_content(chunks[:n])}],
        "max_tokens": 1,
        "stream": False,
        "cache_prompt": False,
    }
    started = time.perf_counter()
    try:
        data = post_json(chat_url, payload, args.timeout)
    except urllib.error.HTTPError as exc:
        print(f"warning: calibration request failed: HTTP {exc.code}: {http_error_text(exc)}; "
              "falling back to the chars-per-token estimate", file=sys.stderr)
        return n
    except Exception as exc:
        print(f"warning: calibration request failed: {exc}; "
              "falling back to the chars-per-token estimate", file=sys.stderr)
        return n
    usage = data.get("usage") or {}
    actual = int(usage.get("prompt_tokens") or 0)
    print(f"calibration prefill: {actual} tokens in {time.perf_counter() - started:.1f}s")
    if actual:
        n = max(1, round(n * isl / actual))
    return n


def run_rep(chat_url, model, content, osl, args):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": osl,
        "stream": False,
        "cache_prompt": False,
    }
    started = time.perf_counter()
    data = post_json(chat_url, payload, args.timeout)
    latency = time.perf_counter() - started
    usage = data.get("usage") or {}
    timings = data.get("timings") or {}
    finish = None
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish = choices[0].get("finish_reason")
    return {
        "isl_actual": int(usage.get("prompt_tokens") or 0),
        "osl_actual": int(usage.get("completion_tokens") or 0),
        "finish_reason": finish,
        "latency_s": round(latency, 1),
        "cache_n": int(timings.get("cache_n") or 0),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "draft_n": int(timings.get("draft_n") or 0),
        "draft_n_accepted": int(timings.get("draft_n_accepted") or 0),
    }


def fmt(value):
    return "n/a" if value is None else f"{value:.2f}"


def print_row(row):
    accept = row["draft_n_accepted"] / row["draft_n"] if row["draft_n"] else None
    print(f"  isl {row['isl_actual']}  osl {row['osl_actual']}  "
          f"finish {row['finish_reason']}  latency {row['latency_s']}s  cache {row['cache_n']}  "
          f"pp {fmt(row['prompt_per_second'])} t/s  tg {fmt(row['predicted_per_second'])} t/s  "
          f"draft {row['draft_n']}  accepted {row['draft_n_accepted']}"
          + (f"  rate {accept:.4f}" if accept is not None else ""), flush=True)


def main():
    # line-buffer stdout so piped runs (tee) keep stdout and stderr in order
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name sent in requests (llama-swap alias)")
    ap.add_argument("--base-url", default="http://fedora-tuf:8080")
    ap.add_argument("--ctx", type=int, default=None,
                    help="max context of the alias; derives isl (85%%) and osl (10%%); "
                         "default: auto-detect from /props (per-slot n_ctx, the cap for a "
                         "single request); explicit value wins, warning on mismatch")
    ap.add_argument("--isl", type=int, default=None,
                    help="target input tokens; overrides --ctx derivation, needs --osl")
    ap.add_argument("--osl", type=int, default=None,
                    help="target output tokens; overrides --ctx derivation, needs --isl")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=14400,
                    help="per-request timeout in seconds; must cover one rep's full "
                         "prefill+decode — the server sends nothing until completion")
    ap.add_argument("--corpus", default=None,
                    help="corpus file path or URL (default: cached Project Gutenberg War and Peace)")
    ap.add_argument("--out", default=None, help="output JSON path (default: results dir, timestamped)")
    args = ap.parse_args()

    if (args.isl is None) != (args.osl is None):
        print("error: --isl and --osl must be given together (or rely on --ctx)", file=sys.stderr)
        return 2

    root = split_base_url(args.base_url)
    chat_url = root + "/v1/chat/completions"
    upstream_prefix = root + "/upstream/" + quote(args.model, safe="/")
    tokenize_urls = [upstream_prefix + "/tokenize", root + "/tokenize"]

    props_info = None
    props_error = None
    print(f"probe:   /props for {args.model} (loads the model if unloaded) ...", flush=True)
    try:
        props_info = props_fields(get_json(upstream_prefix + "/props", WARMUP_TIMEOUT))
    except Exception as exc:
        props_error = exc
    ctx_detected = (props_info or {}).get("n_ctx")
    ctx_source = "flag" if args.ctx is not None else None
    if args.ctx is None:
        if ctx_detected is None:
            if props_error is None:
                detail = "/props response had no usable n_ctx"
            elif isinstance(props_error, urllib.error.HTTPError):
                detail = f"HTTP {props_error.code}: {http_error_text(props_error)}"
            else:
                detail = str(props_error)
            print(f"error: cannot determine the context size ({detail}); pass --ctx "
                  "explicitly (non-llama.cpp backends do not serve /props)", file=sys.stderr)
            return 2
        args.ctx = ctx_detected
        ctx_source = "props"
    elif ctx_detected is not None and ctx_detected != args.ctx:
        slots = (f", total_slots {props_info['total_slots']}"
                 if props_info.get("total_slots") is not None else "")
        print(f"warning: /props reports n_ctx {ctx_detected} (per slot{slots}) for "
              f"{args.model!r}, but --ctx {args.ctx} was given; using --ctx", file=sys.stderr)

    if args.isl is None:
        args.isl = int(args.ctx * 0.85)
        args.osl = int(args.ctx * 0.10)
    if args.isl <= 0 or args.osl <= 0 or args.reps < 1:
        print("error: --isl/--osl must be positive and --reps >= 1", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else (
        RESULTS_DIR / f"{args.model.replace('/', '_')}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )

    print(f"model:   {args.model}")
    if props_info:
        print(f"props:   n_ctx {props_info['n_ctx']}  total_slots {props_info.get('total_slots')}"
              f"  ftype {props_info.get('model_ftype')}")
        print(f"         {props_info.get('model_path')}")
    if args.ctx:
        print(f"ctx:     {args.ctx} ({'via /props' if ctx_source == 'props' else 'from --ctx'})")
    print(f"isl/osl: {args.isl}/{args.osl} target tokens ({args.reps} reps)")
    print(f"api:     {chat_url}")
    print(f"out:     {out} (saved after every rep)")
    print("note: each rep is a cold near-full-context prefill plus a long decode; "
          "expect tens of minutes per rep", flush=True)

    try:
        warmup(chat_url, args.model)
    except urllib.error.HTTPError as exc:
        print(f"error: warmup failed: HTTP {exc.code}: {http_error_text(exc)}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: warmup failed: {exc}", file=sys.stderr)
        return 2

    text = strip_gutenberg(load_corpus(args.corpus))
    chunks = chunk_text(text, CHUNK_CHARS)
    if not chunks:
        print("error: corpus is empty after stripping", file=sys.stderr)
        return 2

    n = calibrate_chunks(chunks, args.isl, tokenize_urls, chat_url, args.model, args)
    if n * args.reps > len(chunks):
        print(f"error: corpus too short: needs {args.reps} x {n} chunks for ~{args.isl}-token reps, "
              f"has {len(chunks)}; use --reps 1 or a longer --corpus", file=sys.stderr)
        return 2

    results = []

    def build_summary():
        ok = [r for r in results if "error" not in r]

        def mean(key):
            vals = [r[key] for r in ok if isinstance(r.get(key), (int, float))]
            return round(sum(vals) / len(vals), 2) if vals else None

        draft_total = sum(r["draft_n"] for r in ok)
        accepted_total = sum(r["draft_n_accepted"] for r in ok)
        return {
            "model": args.model,
            "reps": args.reps,
            "completed": len(ok),
            "failed": len(results) - len(ok),
            "isl_target": args.isl,
            "osl_target": args.osl,
            "avg_prompt_t_s": mean("prompt_per_second"),
            "avg_pred_t_s": mean("predicted_per_second"),
            "avg_latency_s": mean("latency_s"),
            "draft_n": draft_total,
            "accepted": accepted_total,
            "accept_rate": round(accepted_total / draft_total, 4) if draft_total else None,
        }

    def dump_results():
        # write the results so far to the output JSON, atomically, after every rep
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "base_url": root,
                    "model": args.model,
                    "ctx": args.ctx,
                    "ctx_source": ctx_source,
                    "props": props_info,
                    "isl_target": args.isl,
                    "osl_target": args.osl,
                    "reps": args.reps,
                    "corpus": args.corpus or "gutenberg:war_and_peace",
                    "chunks_per_rep": n,
                },
                "summary": build_summary(),
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        os.replace(tmp, out)

    for r in range(args.reps):
        segment = chunks[r * n:(r + 1) * n]
        content = build_content(segment)
        print(f"rep {r + 1}/{args.reps}: prefilling ~{args.isl} tokens, generating up to {args.osl} ...", flush=True)
        try:
            row = run_rep(chat_url, args.model, content, args.osl, args)
            results.append(row)
            print_row(row)
        except urllib.error.HTTPError as exc:
            msg = f"HTTP {exc.code}: {http_error_text(exc)}"
            results.append({"error": msg})
            print(f"  failed: {msg}", file=sys.stderr, flush=True)
        except Exception as exc:
            results.append({"error": str(exc)})
            print(f"  failed: {exc}", file=sys.stderr, flush=True)
        dump_results()

    summary = build_summary()
    ok = [r for r in results if "error" not in r]

    print()
    print(f"summary: {summary['completed']}/{summary['reps']} reps ok  "
          f"pp {fmt(summary['avg_prompt_t_s'])} t/s  tg {fmt(summary['avg_pred_t_s'])} t/s  "
          f"accept_rate {summary['accept_rate'] if summary['accept_rate'] is not None else 'n/a'}")

    print(f"wrote {out}")
    return 0 if ok and len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
