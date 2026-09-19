#!/usr/bin/env python3
"""Generate RP continuations for scenarios in ./scenarios against an
OpenAI-compatible endpoint (llama-swap by default). Standard library only.

Usage:
    python3 rp_generate.py --model <llama-swap-alias> [--scenarios ./scenarios]
                          [--base-url http://fedora-tuf:8080/v1]

By default no sampling parameters are sent, so llama-swap/llama.cpp server
defaults apply — this mirrors real deployment (SillyTavern talks to the same
endpoint). Use --extra '{"temperature": 0.7}' to override.

Output: one JSON per scenario under bench/results/rp/<model>/, containing the
full prompt messages, the continuation, and llama.cpp timings. Transient
transport errors are retried once (HTTP errors are not retried), mirroring
run_reasoning.py.
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
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "rp"


def build_system_prompt(card):
    parts = [
        f"你将扮演「{card.get('name', '角色')}」进行角色扮演。以下是你的角色设定，请严格遵守。",
    ]
    if card.get("description"):
        parts.append(f"\n## 角色描述\n{card['description']}")
    if card.get("personality"):
        parts.append(f"\n## 性格\n{card['personality']}")
    if card.get("scenario"):
        parts.append(f"\n## 场景\n{card['scenario']}")
    if card.get("example_dialogue"):
        parts.append(f"\n## 对话示例\n{card['example_dialogue']}")
    parts.append("\n要求：始终保持角色设定，用中文回复，不要跳出角色，不要以AI助手身份发言。")
    return "\n".join(parts)


def build_messages(scenario):
    system = scenario.get("system_prompt") or build_system_prompt(scenario.get("card", {}))
    messages = [{"role": "system", "content": system}]
    messages.extend(scenario.get("history", []))
    messages.append({"role": "user", "content": scenario["user_message"]})
    return messages


def chat_completion(base_url, payload, timeout):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat_completion_retry(base_url, payload, timeout):
    """chat_completion with one retry on transport errors. HTTPError is a real
    server response, not a transport glitch, so it is not retried."""
    try:
        return chat_completion(base_url, payload, timeout)
    except urllib.error.HTTPError:
        raise
    except OSError as exc:  # URLError, connection reset, timeout, ...
        print(f"    transport error ({exc}); retrying once", flush=True)
        time.sleep(2)
        return chat_completion(base_url, payload, timeout)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name sent in requests (llama-swap alias)")
    ap.add_argument("--base-url", default="http://fedora-tuf:8080/v1")
    ap.add_argument("--scenarios", default=str(SCRIPT_DIR / "scenarios"))
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=None, help="sent only if set")
    ap.add_argument("--extra", default=None, help='extra request fields as JSON, e.g. \'{"temperature": 0.7}\'')
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default=None, help="output dir (default: bench/results/rp/<model>/)")
    args = ap.parse_args()

    scenario_dir = Path(args.scenarios)
    scenarios = sorted(scenario_dir.glob("*.json"))
    if not scenarios:
        print(f"no scenarios found in {scenario_dir}", file=sys.stderr)
        return 2

    try:
        extra = json.loads(args.extra) if args.extra else {}
    except json.JSONDecodeError as exc:
        ap.error(f"--extra is not valid JSON: {exc}")
    out_dir = Path(args.out) if args.out else RESULTS_DIR / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for path in scenarios:
        scenario = json.loads(path.read_text(encoding="utf-8"))
        sid = re.sub(r"[^\w.-]", "_", scenario.get("id") or path.stem)
        if sid.startswith("report"):
            ap.error(f"{path.name}: scenario id {sid!r} starts with 'report' and would "
                     f"collide with rp_judge.py output (it skips report*.json)")
        loaded.append((path, scenario, sid))

    for path, scenario, sid in loaded:
        print(f"[{sid}] generating ...", flush=True)
        payload = {
            "model": args.model,
            "messages": build_messages(scenario),
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        if args.seed is not None:
            payload["seed"] = args.seed
        payload.update(extra)

        started = time.perf_counter()
        try:
            data = chat_completion_retry(args.base_url, payload, args.timeout)
            elapsed = time.perf_counter() - started
            choice = data["choices"][0]
            record = {
                "scenario_id": sid,
                "scenario_file": path.name,
                "model": args.model,
                "messages": payload["messages"],
                "reply": choice["message"].get("content") or "",
                "reasoning": choice["message"].get("reasoning_content"),
                "finish_reason": choice.get("finish_reason"),
                "elapsed_s": round(elapsed, 2),
                "timings": data.get("timings"),
            }
        except Exception as exc:
            record = {"scenario_id": sid, "scenario_file": path.name, "model": args.model,
                      "messages": payload["messages"], "error": str(exc)}
            print(f"    ERROR: {exc}", flush=True)

        out = out_dir / f"{sid}.json"
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"    wrote {out}", flush=True)

    print(f"\ndone. Judge with: python3 rp_judge.py --run {out_dir} --judge-model <model>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
