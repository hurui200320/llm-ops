#!/usr/bin/env python3
"""Run a small deterministic reasoning suite against an OpenAI-compatible endpoint.

Defaults target llama-swap on the LAN. Standard library only.

Usage:
    python3 run_reasoning.py --model <llama-swap-alias> [--difficulty light|hard]
                             [--limit N] [--temperature 0.0] [--max-tokens 16384]
                             [--base-url http://fedora-tuf:8080/v1]

Each problem asks the model to end with "ANSWER: <answer>". Scoring extracts the
last ANSWER: line (fallback: last number in the reply) and compares numerically.
If the reply content is empty, scoring falls back to reasoning_content; that is
deliberate tolerance for the llama.cpp response-in-reasoning bug synbad gates
on (layer 0), not a replacement for it. Transient transport errors are retried
once (HTTP errors are not retried); the summary reports how many requests still
failed. Raw replies and a summary are written to bench/results/reasoning/.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "reasoning"

SYSTEM_PROMPT = (
    "You are a careful reasoner. Solve the problem step by step. "
    "Then output one final line in exactly this format:\n"
    "ANSWER: <final answer>\n"
    "The final answer should be a single number unless the problem says otherwise."
)

ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize(text):
    """Normalize an answer string for comparison. Returns int/float/str."""
    s = text.strip().rstrip(".").strip()
    s = s.replace("$", "").replace(",", "").replace("\\!", "").strip()
    s = s.strip("*` ")
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s.lower()


def answers_match(predicted, expected):
    p, e = normalize(predicted), normalize(expected)
    if isinstance(p, (int, float)) and isinstance(e, (int, float)):
        return abs(float(p) - float(e)) < 1e-6
    return str(p) == str(e)


def extract_answer(content):
    matches = ANSWER_RE.findall(content)
    if matches:
        ans = matches[-1].strip()
        m = NUMBER_RE.search(ans)
        return m.group(0) if m else ans
    numbers = NUMBER_RE.findall(content)
    return numbers[-1] if numbers else None


def chat_completion(base_url, model, problem, args):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        return json.loads(resp.read())


def chat_completion_retry(base_url, model, problem, args):
    """chat_completion with one retry on transport errors. HTTPError is a real
    server response, not a transport glitch, so it is not retried."""
    try:
        return chat_completion(base_url, model, problem, args)
    except urllib.error.HTTPError:
        raise
    except OSError as exc:  # URLError, connection reset, timeout, ...
        print(f"    transport error ({exc}); retrying once", flush=True)
        time.sleep(2)
        return chat_completion(base_url, model, problem, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name sent in requests (llama-swap alias)")
    ap.add_argument("--base-url", default="http://fedora-tuf:8080/v1")
    ap.add_argument("--problems", default=str(SCRIPT_DIR / "problems.jsonl"))
    ap.add_argument("--difficulty", choices=["light", "hard"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="max problems (after difficulty filter)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--timeout", type=float, default=1800, help="per-request timeout in seconds")
    ap.add_argument("--out", default=None, help="output JSON path (default: results dir, timestamped)")
    args = ap.parse_args()

    problems = []
    with open(args.problems, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    if args.difficulty:
        problems = [p for p in problems if p["difficulty"] == args.difficulty]
    if args.limit:
        problems = problems[: args.limit]
    if not problems:
        print("no problems selected", file=sys.stderr)
        return 2

    results = []
    for i, prob in enumerate(problems, 1):
        print(f"[{i}/{len(problems)}] {prob['id']} ({prob['difficulty']}) ...", flush=True)
        started = time.perf_counter()
        try:
            data = chat_completion_retry(args.base_url, args.model, prob["problem"], args)
            elapsed = time.perf_counter() - started
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            if not content:
                content = choice["message"].get("reasoning_content") or ""
            timings = data.get("timings") or {}
            predicted = extract_answer(content)
            correct = predicted is not None and answers_match(predicted, prob["answer"])
            results.append({
                "id": prob["id"],
                "difficulty": prob["difficulty"],
                "expected": prob["answer"],
                "predicted": predicted,
                "correct": correct,
                "elapsed_s": round(elapsed, 2),
                "prompt_per_second": timings.get("prompt_per_second"),
                "predicted_per_second": timings.get("predicted_per_second"),
                "finish_reason": choice.get("finish_reason"),
                "reply": content,
            })
        except Exception as exc:
            results.append({
                "id": prob["id"],
                "difficulty": prob["difficulty"],
                "expected": prob["answer"],
                "predicted": None,
                "correct": False,
                "error": str(exc),
            })
        print(f"    expected {prob['answer']} | got {results[-1]['predicted']} | "
              f"{'ok' if results[-1]['correct'] else 'MISS'}"
              + (f" | {results[-1]['error']}" if "error" in results[-1] else ""), flush=True)

    def stats(rows):
        total = len(rows)
        correct = sum(1 for r in rows if r["correct"])
        errors = sum(1 for r in rows if "error" in r)
        speeds = [r["predicted_per_second"] for r in rows if r.get("predicted_per_second")]
        return {
            "total": total,
            "correct": correct,
            "errors": errors,
            "accuracy": round(correct / total, 4) if total else None,
            "avg_tg_t_s": round(sum(speeds) / len(speeds), 2) if speeds else None,
        }

    summary = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "overall": stats(results),
        "light": stats([r for r in results if r["difficulty"] == "light"]),
        "hard": stats([r for r in results if r["difficulty"] == "hard"]),
    }

    print()
    for key in ("overall", "light", "hard"):
        s = summary[key]
        if s["total"]:
            print(f"{key:8s} {s['correct']}/{s['total']} ({s['accuracy'] * 100:.1f}%)"
                  + (f"  avg tg {s['avg_tg_t_s']} t/s" if s["avg_tg_t_s"] else "")
                  + (f"  [errors: {s['errors']}]" if s["errors"] else ""))

    out = Path(args.out) if args.out else (
        RESULTS_DIR / f"{args.model.replace('/', '_')}-{time.strftime('%Y%m%d-%H%M')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out}")
    if all("error" in r for r in results):
        print("all requests failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
