#!/usr/bin/env python3
"""Run a small deterministic reasoning suite against an OpenAI-compatible endpoint.

Defaults target llama-swap on the LAN. Standard library only.

Usage:
    python3 run_reasoning.py --model <llama-swap-alias> [--difficulty light]
                             [--max-tokens 16384] [--base-url http://fedora-tuf:8080/v1]
                             [--problems problems.jsonl ../.cache/reasoning/frontier.jsonl]

The suite has three families: light sanity math (problems.jsonl), AIME 2026 and
generated zebra logic puzzles (both built by fetch_frontier.py into
bench/.cache/reasoning/frontier.jsonl — pass both files to --problems to run
the full suite). Sampling is left at the server default on purpose: the goal
is performance under the deployed setup, not a leaderboard number.

Each numeric problem asks the model to end with "ANSWER: <answer>". Scoring
extracts the last ANSWER: line (fallback: last number in the reply) and
compares numerically. Grid problems (zebra) ask for a final "SOLUTION:" block,
one "|"-separated line per house; scoring compares cells (case/whitespace
insensitive), reports puzzle-level pass/fail plus a cell-level partial score.

If the reply content is empty, scoring falls back to reasoning_content; that is
deliberate tolerance for the llama.cpp response-in-reasoning bug synbad gates
on (layer 0), not a replacement for it. Transient transport errors and gateway
5xx (a restarting llama-swap answers 500/502 instantly) get a single retry
after a 60s wait — a cold 31B load takes minutes, but a problem that burns
the full 30-minute timeout twice is a genuine fail, not a transient; 4xx
stays fatal, and the run aborts after 5 consecutive errors. The summary
reports how many requests still failed, plus a GATE verdict line (PASS /
REVIEW / FAIL on the gate subset: l12 must pass, then >=6/9 scored to pass).
Raw replies and a summary are written to bench/results/reasoning/.
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

GRID_SYSTEM_PROMPT = (
    "You are a careful logician. Solve the puzzle step by step by pure deduction. "
    "Then output the completed grid as a final block in exactly this format:\n"
    "SOLUTION:\n"
    "1 | <value> | <value> | ...\n"
    "2 | <value> | <value> | ...\n"
    "One line per house, in house order (house number first), with one value per "
    "characteristic in the order the puzzle lists them. Use the exact values from "
    "the puzzle. Nothing after the block."
)

ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
SOLUTION_RE = re.compile(r"^\s*\**\s*SOLUTION\s*\**\s*:?\s*(.*)$", re.IGNORECASE)
SEPARATOR_CELL_RE = re.compile(r"^[-: ]*$")
HOUSE_LABEL_RE = re.compile(r"^house\s*#?\s*\d+$")

# A restarting llama-swap/llamacpp answers instantly with 500/502, and a cold
# 31B load takes minutes — so one retry after a 60s wait. A single retry is
# deliberate: a problem that burns the full 30-minute timeout twice is a
# genuine fail (model rambling past budget), not a transient, and on a
# single-slot setup each extra retry is another half hour of wall time.
# 4xx is a real error and stays fatal.
RETRY_WAITS_S = (60,)
TRANSIENT_HTTP_CODES = (500, 502, 503, 504)
CONSECUTIVE_ERROR_ABORT = 5


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


def normalize_cell(cell):
    s = cell.strip().strip("`").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".").rstrip(",")
    return s.lower()


def parse_solution_grid(content):
    """Parse the final SOLUTION: block into rows of cell strings.

    Returns a list of rows, each row a list of normalized cells with the
    leading house number removed when present. Markdown table separators and
    border pipes are tolerated; rows without pipes are ignored.
    """
    lines = content.splitlines()
    start = None
    for i, line in enumerate(lines):
        m = SOLUTION_RE.match(line.strip())
        if m:
            start = (i, m.group(1))
    if start is None:
        return []
    rows = []
    first_idx, inline = start
    candidates = [inline] if inline.strip() else []
    candidates += lines[first_idx + 1:]
    for line in candidates:
        if "|" not in line:
            continue
        cells = [normalize_cell(c) for c in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if not cells or all(SEPARATOR_CELL_RE.match(c) for c in cells):
            continue
        if cells[0].isdigit() or HOUSE_LABEL_RE.match(cells[0]):
            cells = cells[1:]
        if cells:
            rows.append(cells)
    return rows


def score_grid(predicted_rows, answer):
    """Score parsed rows against the expected grid.

    Returns (all_correct, cell_accuracy). Rows are compared positionally in
    house order; missing rows or cells count as wrong. Extra trailing rows
    (prose with pipes, summary lines) do not fail an otherwise perfect grid —
    they stay visible in the stored "predicted" field. Predicted rows that
    merely repeat the header labels (markdown table heads) are dropped first.
    """
    header_cells = [normalize_cell(h) for h in answer["header"]]
    predicted_rows = [r for r in predicted_rows if r != header_cells]
    expected = [row[1:] for row in answer["rows"]]
    n_cells = sum(len(row) for row in expected)
    matched = 0
    all_correct = True
    for i, exp_row in enumerate(expected):
        pred_row = predicted_rows[i] if i < len(predicted_rows) else []
        for j, exp_cell in enumerate(exp_row):
            pred_cell = pred_row[j] if j < len(pred_row) else None
            if pred_cell is not None and pred_cell == normalize_cell(exp_cell):
                matched += 1
            else:
                all_correct = False
    return all_correct, (matched / n_cells if n_cells else 0.0)


def system_prompt_for(prob):
    if prob.get("answer_kind") == "grid":
        return GRID_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def chat_completion(base_url, model, prob, args):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt_for(prob)},
            {"role": "user", "content": prob["problem"]},
        ],
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


def chat_completion_retry(base_url, model, prob, args):
    """chat_completion with backoff retries on transport errors and transient
    HTTP 5xx (500/502/503/504 — what a restarting gateway answers instantly).
    Other HTTP errors (4xx) are real server responses and stay fatal."""
    last_exc = None
    describe = ""
    for attempt, wait in enumerate((0,) + RETRY_WAITS_S):
        if wait:
            print(f"    transient error ({describe}); waiting {wait}s "
                  f"before retry {attempt}/{len(RETRY_WAITS_S)}", flush=True)
            time.sleep(wait)
        try:
            return chat_completion(base_url, model, prob, args)
        except urllib.error.HTTPError as exc:
            if exc.code not in TRANSIENT_HTTP_CODES:
                raise
            last_exc, describe = exc, f"HTTP {exc.code}"
        except OSError as exc:  # URLError, connection reset, timeout, ...
            last_exc, describe = exc, str(exc)
    raise last_exc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name sent in requests (llama-swap alias)")
    ap.add_argument("--base-url", default="http://fedora-tuf:8080/v1")
    ap.add_argument("--problems", nargs="+", default=[str(SCRIPT_DIR / "problems.jsonl")],
                    help="problem jsonl files (e.g. problems.jsonl ../.cache/reasoning/frontier.jsonl)")
    ap.add_argument("--difficulty", choices=["light", "frontier"], default=None)
    ap.add_argument("--family", choices=["light", "aime", "zebra"], default=None)
    ap.add_argument("--skip", default=None,
                    help="comma-separated problem ids to exclude from this run "
                         "(typo guard: warns when an id matches no problem)")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--timeout", type=float, default=1800, help="per-request timeout in seconds")
    ap.add_argument("--out", default=None, help="output JSON path (default: results dir, timestamped)")
    args = ap.parse_args()

    problems = []
    for path in args.problems:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
    if args.difficulty:
        problems = [p for p in problems if p["difficulty"] == args.difficulty]
    if args.family:
        problems = [p for p in problems if (p.get("family") or p.get("difficulty")) == args.family]
    if args.skip:
        skip_ids = {s.strip() for s in args.skip.split(",") if s.strip()}
        all_ids = {p["id"] for p in problems}
        problems = [p for p in problems if p["id"] not in skip_ids]
        unknown = skip_ids - all_ids
        if unknown:
            print(f"warning: --skip ids matching no loaded problem: {', '.join(sorted(unknown))}",
                  file=sys.stderr)
    if not problems:
        print("no problems selected", file=sys.stderr)
        return 2

    results = []
    consecutive_errors = 0
    aborted = False
    for i, prob in enumerate(problems, 1):
        family = prob.get("family") or prob.get("difficulty")
        print(f"[{i}/{len(problems)}] {prob['id']} ({family}) ...", flush=True)
        started = time.perf_counter()
        try:
            data = chat_completion_retry(args.base_url, args.model, prob, args)
            elapsed = time.perf_counter() - started
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            if not content:
                content = choice["message"].get("reasoning_content") or ""
            timings = data.get("timings") or {}
            usage = data.get("usage") or {}
            row = {
                "id": prob["id"],
                "family": family,
                "difficulty": prob["difficulty"],
                "expected": prob["answer"],
            }
            if prob.get("answer_kind") == "grid":
                predicted_rows = parse_solution_grid(content)
                correct, cell_accuracy = score_grid(predicted_rows, prob["answer"])
                row.update({
                    "predicted": predicted_rows,
                    "correct": correct,
                    "cell_accuracy": round(cell_accuracy, 4),
                })
            else:
                predicted = extract_answer(content)
                correct = predicted is not None and answers_match(predicted, prob["answer"])
                row.update({
                    "predicted": predicted,
                    "correct": correct,
                })
            row.update({
                "elapsed_s": round(elapsed, 2),
                "prompt_per_second": timings.get("prompt_per_second"),
                "predicted_per_second": timings.get("predicted_per_second"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": choice.get("finish_reason"),
                "reply": content,
            })
            if prob.get("answer_kind") == "grid":
                extra = f" | cells {row['cell_accuracy']:.0%}"
            else:
                extra = ""
            results.append(row)
            consecutive_errors = 0
        except Exception as exc:
            results.append({
                "id": prob["id"],
                "family": prob.get("family") or prob.get("difficulty"),
                "difficulty": prob["difficulty"],
                "expected": prob["answer"],
                "predicted": None,
                "correct": False,
                "error": str(exc),
            })
            extra = ""
            consecutive_errors += 1
        print(f"    expected {prob['answer'] if not isinstance(prob['answer'], dict) else '<grid>'} | "
              f"{'ok' if results[-1]['correct'] else 'MISS'}{extra}"
              + (f" | {results[-1]['error']}" if "error" in results[-1] else ""), flush=True)
        if consecutive_errors >= CONSECUTIVE_ERROR_ABORT:
            print(f"aborting: {CONSECUTIVE_ERROR_ABORT} consecutive errors — the serving stack is "
                  "likely down, not transient; results so far are still written", file=sys.stderr)
            aborted = True
            break

    def stats(rows):
        total = len(rows)
        correct = sum(1 for r in rows if r["correct"])
        errors = sum(1 for r in rows if "error" in r)
        speeds = [r["predicted_per_second"] for r in rows if r.get("predicted_per_second")]
        cell_accs = [r["cell_accuracy"] for r in rows if "cell_accuracy" in r]
        return {
            "total": total,
            "correct": correct,
            "errors": errors,
            "accuracy": round(correct / total, 4) if total else None,
            "avg_tg_t_s": round(sum(speeds) / len(speeds), 2) if speeds else None,
            "avg_cell_accuracy": round(sum(cell_accs) / len(cell_accs), 4) if cell_accs else None,
        }

    families = []
    for r in results:
        if r["family"] not in families:
            families.append(r["family"])
    summary = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "overall": stats(results),
        "families": {fam: stats([r for r in results if r["family"] == fam]) for fam in families},
    }

    print()
    s = summary["overall"]
    print(f"{'overall':8s} {s['correct']}/{s['total']} ({s['accuracy'] * 100:.1f}%)"
          + (f"  avg tg {s['avg_tg_t_s']} t/s" if s["avg_tg_t_s"] else "")
          + (f"  [errors: {s['errors']}]" if s["errors"] else ""))
    for fam, s in summary["families"].items():
        if s["total"]:
            print(f"{fam:8s} {s['correct']}/{s['total']} ({s['accuracy'] * 100:.1f}%)"
                  + (f"  cells {s['avg_cell_accuracy'] * 100:.1f}%" if s["avg_cell_accuracy"] is not None else "")
                  + (f"  avg tg {s['avg_tg_t_s']} t/s" if s["avg_tg_t_s"] else "")
                  + (f"  [errors: {s['errors']}]" if s["errors"] else ""))

    gate_ids = {"l12", "aime26-10", "aime26-11", "aime26-15",
                "zebra-5x5-01", "zebra-5x5-02", "zebra-5x5-03",
                "zebra-6x4-01", "zebra-6x4-02", "zebra-6x6-01"}
    gate_rows = [r for r in results if r["id"] in gate_ids]
    if len(gate_rows) == len(gate_ids):
        sanity = next(r for r in gate_rows if r["id"] == "l12")
        scored = [r for r in gate_rows if r["id"] != "l12"]
        n_ok = sum(1 for r in scored if r["correct"])
        if not sanity["correct"]:
            verdict = "FAIL (sanity l12 failed — harness broken, not a model verdict)"
        elif n_ok >= 6:
            verdict = "PASS"
        elif n_ok == 5:
            verdict = "REVIEW"
        else:
            verdict = "FAIL"
        print(f"GATE {verdict} (sanity {'ok' if sanity['correct'] else 'MISS'}, "
              f"scored {n_ok}/9)")
        summary["gate"] = {"verdict": verdict.split(" ")[0],
                           "sanity_ok": sanity["correct"],
                           "scored_correct": n_ok, "scored_total": len(scored)}

    out = Path(args.out) if args.out else (
        RESULTS_DIR / f"{args.model.replace('/', '_')}-{time.strftime('%Y%m%d-%H%M')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out}")
    if aborted:
        print("run aborted early (see above); results are incomplete", file=sys.stderr)
        return 1
    if all("error" in r for r in results):
        print("all requests failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
