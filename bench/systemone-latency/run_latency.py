#!/usr/bin/env python3
"""Sequential /v1/systemone latency benchmark; never manages deployments."""

import argparse
import base64
import hashlib
import http.client
import json
import math
import mimetypes
import os
import random
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "systemone"
QUESTIONS = {
    "next_action": {
        "type": "choice",
        "instructions": (
            "Choose the next action from the current agent status at the end of the state. "
            "Earlier records are background, not the current status."
        ),
        "criteria": {
            "inspect_logs": "Investigate the failed test before editing anything else.",
            "run_tests": "Run tests after an untested code change.",
            "edit_file": "Apply an already diagnosed code fix.",
            "read_file": "Read source code to understand its behavior.",
            "search_code": "Find the relevant implementation in the repository.",
            "save_file": "Save an unsaved editor buffer.",
            "open_page": "Navigate to a requested web page.",
            "click_button": "Activate the identified UI button.",
            "type_text": "Enter requested text into a focused input.",
            "scroll_page": "Reveal content below the visible viewport.",
            "wait": "Wait for an operation that is still running.",
            "retry": "Retry a confirmed transient failure.",
            "ask_user": "Ask for a missing requirement or authorization.",
            "summarize": "Summarize completed work for the user.",
            "finish": "Stop after verified successful completion.",
            "cancel": "Cancel an operation the user no longer wants.",
        },
    }
}
STATUS = "\nCurrent agent status: The latest test failed. Its cause is unknown. Inspect the test logs next."


class Client:
    """One persistent connection, no automatic retries (including stale sockets)."""

    def __init__(self, base_url, timeout, api_key=None):
        parsed = urlsplit(base_url.rstrip("/"))
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("base URL must be HTTP(S), without credentials, query or fragment")
        connection = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        self.connection = connection(parsed.hostname, parsed.port, timeout=timeout)
        self.root = parsed.path.rstrip("/")
        if self.root.endswith("/v1"):
            self.root = self.root[:-3]
        self.headers = {"Content-Type": "application/json", "User-Agent": "llm-ops-systemone/1.0"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key

    def request(self, method, path, body=None):
        started = time.perf_counter_ns()
        try:
            self.connection.request(method, self.root + path, body=body, headers=self.headers)
            response = self.connection.getresponse()
            raw = response.read()
            elapsed = (time.perf_counter_ns() - started) / 1_000_000
            return response.status, raw, elapsed
        except (OSError, http.client.HTTPException):
            self.connection.close()
            raise

    def get(self, path):
        status, raw, _ = self.request("GET", path)
        if status != 200:
            raise ValueError(f"GET {path}: HTTP {status}")
        return raw

    def close(self):
        self.connection.close()


def encode_payload(model, state, image=None):
    payload = {"model": model, "state": state, "questions": QUESTIONS}
    if image is not None:
        payload["images"] = [image]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def make_state(chars, seed, identity):
    rng = random.Random(seed)
    records = []
    size = 0
    while size < chars:
        record = (
            f"Record {len(records)}: module {rng.choice(('parser', 'router', 'editor', 'storage', 'client'))}; "
            f"operation {rng.choice(('read', 'validate', 'update', 'render', 'save'))}; "
            f"result {rng.choice(('completed', 'pending', 'reviewed', 'unchanged'))}; "
            f"item {rng.randrange(10000)}. "
            f"The agent checked {rng.choice(('input fields', 'file paths', 'response headers', 'test fixtures'))} "
            "and recorded the observation for later review.\n"
        )
        records.append(record)
        size += len(record)
    # The identity changes at the start of the state, not just at its suffix.
    return f"Independent observation {identity}\n" + "".join(records)[:chars] + STATUS


def validate_response(data):
    if not isinstance(data, dict):
        raise ValueError("response must be an object")
    answer = data.get("answers", {}).get("next_action")
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError("missing next_action choice answer")
    options = QUESTIONS["next_action"]["criteria"]
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(options):
        raise ValueError("probability keys do not match options")
    values = list(probabilities.values())
    if any(isinstance(p, bool) or not isinstance(p, (int, float))
           or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("invalid probability")
    if not math.isclose(sum(values), 1.0, abs_tol=0.001):
        raise ValueError("probabilities do not sum to one")
    choice = answer.get("choice")
    if choice not in options or probabilities[choice] < max(values) - 0.000001:
        raise ValueError("choice is not a highest-probability option")
    usage = data.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("missing usage")
    tokens = usage.get("input_tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise ValueError("invalid input_tokens")
    if usage.get("output_tokens") != 0:
        raise ValueError("decision output_tokens must be zero")
    return tokens, choice


def measure(client, body):
    started = time.perf_counter_ns()
    row = {"ok": False, "http_status": None, "input_tokens": None}
    try:
        status, raw, elapsed = client.request("POST", "/v1/systemone", body)
        row.update(http_status=status, latency_ms=elapsed, response_bytes=len(raw))
        if status != 200:
            raise ValueError(f"HTTP {status}")
        tokens, choice = validate_response(json.loads(raw))
        row.update(ok=True, input_tokens=tokens, choice=choice, expected_choice=choice == "inspect_logs")
    except (OSError, http.client.HTTPException, ValueError, TypeError, AttributeError) as exc:
        row.setdefault("latency_ms", (time.perf_counter_ns() - started) / 1_000_000)
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def percentile(values, percent):
    """Linear interpolation between sorted samples, including endpoints."""
    ordered = sorted(values)
    pos = (len(ordered) - 1) * percent / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def distribution(values):
    if not values:
        return None
    return {
        "mean": statistics.mean(values), "stdev": statistics.pstdev(values),
        "min": min(values), "p50": percentile(values, 50),
        "p95": percentile(values, 95), "max": max(values),
    }


def summarize(rows):
    valid = [row for row in rows if row["ok"]]
    return {
        "requests": len(rows), "successful": len(valid), "errors": len(rows) - len(valid),
        "latency_ms": distribution([row["latency_ms"] for row in valid]),
        "failed_latency_ms": distribution([row["latency_ms"] for row in rows if not row["ok"]]),
        "input_tokens": distribution([row["input_tokens"] for row in valid]),
        "expected_choice_count": sum(row["expected_choice"] for row in valid),
    }


def calibrate(client, model, target, seed, run_id):
    chars = target * 4
    history = []
    for attempt in range(6):
        state = make_state(chars, seed, f"{run_id}-calibration-{target}-{attempt}")
        row = measure(client, encode_payload(model, state))
        history.append({"state_chars": chars, **row})
        if not row["ok"]:
            raise ValueError(f"calibration failed: {row['error']}")
        actual = row["input_tokens"]
        if abs(actual - target) <= max(16, target * 0.02):
            return chars, history
        # Adjust the variable-length background, accounting for fixed template/options.
        next_chars = max(0, chars + round((target - actual) * chars / max(1, actual)))
        if next_chars == chars:
            break
        chars = next_chars
    best = min(history, key=lambda row: abs(row["input_tokens"] - target))
    return best["state_chars"], history


def read_metrics(client, path):
    try:
        raw = client.get(path).decode("utf-8")
        values = {}
        for line in raw.splitlines():
            if line.startswith(("llamacpp:prompt_tokens_total ", "llamacpp:prompt_tokens_cached_total ")):
                key, value = line.split()
                values[key] = float(value)
        if len(values) != 2:
            raise ValueError("prompt metrics unavailable")
        return {"values": values}
    except (OSError, http.client.HTTPException, ValueError) as exc:
        return {"error": str(exc)}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("LLAMA_SWAP_URL", "http://fedora-tuf:8080"))
    parser.add_argument("--api-key-env", default="LLAMA_SWAP_KEY", help="environment variable containing auth key")
    parser.add_argument("--label", default="", help="human-supplied deployment description, e.g. tensor-ub1024")
    parser.add_argument("--input-tokens", default="1024,4096", help="approximate total text prompt sizes")
    parser.add_argument("--reps", type=positive_int, default=50, help="measured requests per workload")
    parser.add_argument("--warmup", type=nonnegative_int, default=5, help="untimed requests per workload")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-mode", choices=("fresh", "reuse"), default="fresh")
    parser.add_argument("--image", type=Path, action="append", default=[], help="add text+image workloads (one image per request)")
    parser.add_argument("--timeout", type=float, default=120, help="socket timeout in seconds, not an overall deadline")
    parser.add_argument("--probe-prefix", help="metadata path prefix; default /upstream/<model>; use empty string for direct server")
    parser.add_argument("--output-dir", type=Path, help="new artifact directory; default bench/results/systemone/<timestamp>-<id>")
    args = parser.parse_args(argv)
    try:
        targets = [int(value) for value in args.input_tokens.split(",")]
        if not targets or any(value <= 0 for value in targets) or len(set(targets)) != len(targets):
            raise ValueError("input token sizes must be distinct positive integers")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        client = Client(args.base_url, args.timeout, os.environ.get(args.api_key_env))
        images = []
        for path in args.image:
            mime = mimetypes.guess_type(path.name)[0]
            if not mime or not mime.startswith("image/"):
                raise ValueError(f"unrecognized image extension: {path.name}")
            raw = path.read_bytes()
            images.append(("data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii"),
                           {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "mime": mime}))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    run_id = uuid.uuid4().hex
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or RESULTS_DIR / f"{stamp}-{run_id[:8]}"
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        client.close()
        parser.error(f"cannot create artifact directory: {exc}")
    prefix = args.probe_prefix if args.probe_prefix is not None else "/upstream/" + quote(args.model, safe="")
    prefix = prefix.rstrip("/")
    metadata = {
        "started_at": stamp, "run_id": run_id, "model": args.model, "base_url": args.base_url,
        "label": args.label, "seed": args.seed, "state_mode": args.state_mode,
        "reps": args.reps, "warmup": args.warmup, "input_token_targets": targets,
        "images": [info for _, info in images], "timeout_s": args.timeout,
        "concurrency": 1, "percentile_method": "linear interpolation, (n-1)*p",
        "timing": "prepared HTTP request send to full body read; excludes JSON parsing; no retries",
        "cache_note": "fresh states minimize reuse, not guaranteed cold; cache_prompt is not sent",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    rows = []
    workloads = []
    status = "failed"
    metrics_before = None
    exit_code = 1
    try:
        try:
            props = json.loads(client.get(prefix + "/props"))
            metadata["props"] = {key: props[key] for key in (
                "build_info", "total_slots", "model_alias", "model_ftype", "is_sleeping"
            ) if key in props}
            metadata["props"]["n_ctx"] = props.get("default_generation_settings", {}).get("n_ctx")
        except (OSError, http.client.HTTPException, ValueError, AttributeError, TypeError) as exc:
            metadata["props_error"] = str(exc)
        for target in targets:
            chars, history = calibrate(client, args.model, target, args.seed, run_id)
            closest = min(history, key=lambda row: abs(row["input_tokens"] - target))
            if abs(closest["input_tokens"] - target) > max(16, target * 0.02):
                print(f"warning: text-{target} calibration reached {closest['input_tokens']} tokens; "
                      "inspect actual measured usage", file=sys.stderr)
            for image_index in range(len(images) + 1):
                workload = {
                    "name": f"text-{target}" + (f"-image{image_index}" if image_index else ""),
                    "text_target_tokens": target, "background_chars": chars,
                    "image_index": image_index, "calibration": history,
                }
                workloads.append(workload)
            print(f"calibrated text-{target}: {chars} background chars", flush=True)

        def prepare(workload, phase, index):
            reused = args.state_mode == "reuse"
            identity = f"{run_id}-{workload['name']}-" + ("shared" if reused else f"{phase}-{index}")
            seed = args.seed if reused else args.seed + index
            state = make_state(workload["background_chars"], seed, identity)
            image = images[workload["image_index"] - 1][0] if workload["image_index"] else None
            return encode_payload(args.model, state, image)

        for workload in workloads:
            warmups = []
            for index in range(args.warmup):
                row = measure(client, prepare(workload, "warmup", index))
                warmups.append(row)
                if not row["ok"]:
                    workload["warmup_results"] = warmups
                    raise ValueError(f"{workload['name']} warmup failed: {row['error']}")
            workload["warmup_results"] = warmups
            if warmups:
                print(f"{workload['name']} warmup ms: " +
                      ", ".join(f"{row['latency_ms']:.1f}" for row in warmups), flush=True)
        metrics_before = read_metrics(client, prefix + "/metrics")
        print(f"measuring {args.reps} requests per workload; artifacts: {output}", flush=True)
        rng = random.Random(args.seed)
        with (output / "requests.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(args.reps):
                order = list(workloads)
                rng.shuffle(order)
                for workload in order:
                    body = prepare(workload, "measured", index)
                    row = {"workload": workload["name"], "rep": index, **measure(client, body)}
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    if not row["ok"]:
                        print(f"{workload['name']} rep {index}: {row['error']}", file=sys.stderr)
        status = "completed"
        exit_code = int(any(not row["ok"] for row in rows))
    except KeyboardInterrupt:
        status = "interrupted"
        exit_code = 130
    except (OSError, http.client.HTTPException, ValueError) as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        print(metadata["error"], file=sys.stderr)
    finally:
        metrics_after = read_metrics(client, prefix + "/metrics") if metrics_before else None
        metadata.update(status=status, workloads=workloads, metrics_before=metrics_before, metrics_after=metrics_after)
        if metrics_before and metrics_after and "values" in metrics_before and "values" in metrics_after:
            metadata["measured_prompt_metrics_delta"] = {
                key: value - metrics_before["values"][key] for key, value in metrics_after["values"].items()
            }
        summaries = {w["name"]: summarize([r for r in rows if r["workload"] == w["name"]]) for w in workloads}
        (output / "summary.json").write_text(json.dumps({"metadata": metadata, "workloads": summaries}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        client.close()
    print("workload                  ok/errors     mean ms     p50 ms     p95 ms   input p50  expected")
    for name, summary in summaries.items():
        latency = summary["latency_ms"]
        if latency:
            print(f"{name:25} {summary['successful']:4}/{summary['errors']:<6} "
                  f"{latency['mean']:10.2f} {latency['p50']:10.2f} {latency['p95']:10.2f} "
                  f"{summary['input_tokens']['p50']:11.0f} "
                  f"{summary['expected_choice_count']}/{summary['successful']}")
        else:
            print(f"{name:25} {summary['successful']:4}/{summary['errors']:<6} no valid responses")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
