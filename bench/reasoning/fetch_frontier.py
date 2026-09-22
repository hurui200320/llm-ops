#!/usr/bin/env python3
"""Build the frontier tier of the reasoning suite.

Two sources, one output file (bench/.cache/reasoning/frontier.jsonl):

- AIME 2026 problems are fetched from MathArena/aime_2026 via the HuggingFace
  datasets-server REST API (plain urllib, no deps). The problems themselves
  stay in the gitignored cache; only ids + answer hashes are committed in
  frontier_manifest.json. MAA-copyrighted text must never be committed.
- Zebra (logic-grid) puzzles are generated here: a random solution grid, the
  full set of clues true in it, then clue minimization under a
  uniqueness-checking CSP solver (the same construction ZebraLogicBench uses;
  the public release hides its answers, so we make our own — fresh numbers,
  no contamination, no license questions).

Deterministic for a given --seed (and upstream dataset revision for AIME):
rerunning with the same arguments rebuilds the same set; frontier_manifest.json
pins it. --check verifies the cached file still matches the manifest.

Usage:
    python3 fetch_frontier.py [--aime-count 15] [--zebra-spec "3x4:2,4x4:5,4x5:3,5x5:3,6x4:2,6x6:1"]
                              [--seed 20260920] [--out PATH] [--check]

Zebra spec format: "NxM:count,...". N = houses (rows), M = characteristics.
Difficulty tracks the search space (N!)^M (not the N*M product: 6x4
out-searches 5x5, and 6x6 at ~1.4e17 is a category of its own — the gate
spec carries one 6x6 as a near-impossible anchor). If every model passes,
move the spec to bigger sizes and --aime-count 30; if every model fails,
first rule out max-tokens and format-compliance issues in the run logs,
then move the spec to smaller sizes. Any argument change re-freezes the
manifest — calibrate on the strongest model, freeze once, then compare
all models paired (see bench/README.md §3 for the full workflow).
"""

import argparse
import hashlib
import itertools
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR.parent / ".cache" / "reasoning"
DEFAULT_OUT = CACHE_DIR / "frontier.jsonl"
MANIFEST_PATH = SCRIPT_DIR / "frontier_manifest.json"

USER_AGENT = "llm-ops-bench/1.0"
AIME_DATASET = "MathArena/aime_2026"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
HF_API = "https://huggingface.co/api/datasets"

# ---------------------------------------------------------------------------
# AIME fetch
# ---------------------------------------------------------------------------


def http_get_json(url, timeout=60, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except OSError as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise SystemExit(f"error: fetch failed for {url}: {last}")


def fetch_aime(count):
    """Return (problems, revision). problems are jsonl-ready dicts."""
    meta = http_get_json(f"{HF_API}/{AIME_DATASET}")
    revision = meta.get("sha")
    splits = http_get_json(
        f"{DATASETS_SERVER}/splits?dataset={quote(AIME_DATASET, safe='')}"
    )["splits"]
    cfg = splits[0]
    rows = http_get_json(
        f"{DATASETS_SERVER}/rows?dataset={quote(AIME_DATASET, safe='')}"
        f"&config={cfg['config']}&split={cfg['split']}&offset=0&length=100"
    )["rows"]
    if len(rows) < count:
        raise SystemExit(f"error: only {len(rows)} AIME rows available, need {count}")
    problems = []
    for row in rows[:count]:
        r = row["row"]
        problems.append({
            "id": f"aime26-{int(r['problem_idx']):02d}",
            "family": "aime",
            "difficulty": "frontier",
            "answer_kind": "number",
            "problem": r["problem"],
            "answer": str(r["answer"]),
        })
    return problems, revision, len(rows)


# ---------------------------------------------------------------------------
# Zebra puzzle generation (CSP with uniqueness verification)
# ---------------------------------------------------------------------------
# Each attribute: (label for the background list, phrase used in referring
# expressions, value pool). Pools are larger than any grid so value sets vary.

ATTRS = [
    ("name", "is named",
     ["Alice", "Bob", "Carol", "Diana", "Eric", "Frank", "Grace", "Henry"]),
    ("nationality", "is",
     ["Norwegian", "German", "Danish", "British", "Swedish", "Finnish", "Icelandic", "Dutch"]),
    ("favorite color", "likes the color",
     ["red", "green", "blue", "yellow", "white", "black", "purple", "orange"]),
    ("pet", "keeps a",
     ["cat", "dog", "bird", "horse", "fish", "rabbit", "hamster", "snake"]),
    ("favorite drink", "drinks",
     ["tea", "coffee", "milk", "water", "juice", "soda", "lemonade", "cocoa"]),
    ("favorite food", "eats",
     ["pizza", "spaghetti", "stir fry", "stew", "grilled cheese", "salad", "sushi", "pancakes"]),
    ("occupation", "works as",
     ["artist", "engineer", "teacher", "doctor", "lawyer", "nurse", "chef", "farmer"]),
    ("favorite book genre", "enjoys reading",
     ["fantasy", "science fiction", "mystery", "romance", "biography", "history", "poetry", "horror"]),
    ("favorite music genre", "listens to",
     ["pop", "rock", "jazz", "classical", "hip hop", "country", "blues", "electronic"]),
    ("hobby", "enjoys",
     ["gardening", "painting", "chess", "hiking", "photography", "knitting", "cycling", "fishing"]),
]

# Clue kinds and their position predicates (houses 0-based):
#   found:   pos(x) == h          not_at: pos(x) != h
#   same:    pos(x) == pos(y)     not_same: pos(x) != pos(y)
#   dleft:   pos(x) + 1 == pos(y)
#   left:    pos(x) < pos(y)      right:  pos(x) > pos(y)
#   adj:     |pos(x) - pos(y)| == 1
#   gap2:    |pos(x) - pos(y)| == 2   (exactly one house between)
#   gap3:    |pos(x) - pos(y)| == 3   (exactly two houses between)
# x, y are (attr_index, value_index) pairs.

VOWEL_INITIAL = re.compile(r"^[aeiou]")


def article(value):
    return "an" if VOWEL_INITIAL.match(value) else "a"


def cap_first(s):
    return s[:1].upper() + s[1:]


def refer(chosen, attr_idx, val_idx, values):
    label, phrase, pool = ATTRS[chosen[attr_idx]]
    value = values[attr_idx][val_idx]
    if phrase == "works as":
        return f"the person who works as {article(value)} {value}"
    return f"the person who {phrase} {value}"


def render_clue(clue, chosen, values):
    kind = clue[0]
    if kind == "found":
        _, x, h = clue
        return f"{cap_first(refer(chosen, *x, values))} is in house {h + 1}."
    if kind == "not_at":
        _, x, h = clue
        return f"{cap_first(refer(chosen, *x, values))} does not live in house {h + 1}."
    _, x, y = clue
    rx, ry = refer(chosen, *x, values), refer(chosen, *y, values)
    if kind == "same":
        return f"{cap_first(rx)} and {ry} are the same person."
    if kind == "not_same":
        return f"{cap_first(rx)} is not {ry}."
    if kind == "dleft":
        return f"{cap_first(rx)} lives directly to the left of {ry}."
    if kind == "left":
        return f"{cap_first(rx)} lives somewhere to the left of {ry}."
    if kind == "right":
        return f"{cap_first(rx)} lives somewhere to the right of {ry}."
    if kind == "adj":
        return f"{cap_first(rx)} lives next to {ry}."
    if kind == "gap2":
        return f"There is exactly one house between {rx} and {ry}."
    if kind == "gap3":
        return f"There are exactly two houses between {rx} and {ry}."
    raise ValueError(kind)


def clue_holds(clue, pos):
    """pos: dict attr -> tuple(house of each value). Houses 0-based."""
    kind = clue[0]
    if kind == "found":
        _, x, h = clue
        return pos[x[0]][x[1]] == h
    if kind == "not_at":
        _, x, h = clue
        return pos[x[0]][x[1]] != h
    _, x, y = clue
    p1, p2 = pos[x[0]][x[1]], pos[y[0]][y[1]]
    if kind == "same":
        return p1 == p2
    if kind == "not_same":
        return p1 != p2
    if kind == "dleft":
        return p1 + 1 == p2
    if kind == "left":
        return p1 < p2
    if kind == "right":
        return p1 > p2
    if kind == "adj":
        return abs(p1 - p2) == 1
    if kind == "gap2":
        return abs(p1 - p2) == 2
    if kind == "gap3":
        return abs(p1 - p2) == 3
    raise ValueError(kind)


class BudgetExceeded(Exception):
    pass


def count_solutions(n, m, constraints, max_solutions=2, node_budget=200000):
    """Count assignments (one permutation of houses per attribute) satisfying
    all constraints, aborting at max_solutions. Raises BudgetExceeded if the
    search exceeds node_budget (caller treats that as "cannot prove").
    """
    within = [[] for _ in range(m)]
    pair = {}
    for clue in constraints:
        kind = clue[0]
        if kind in ("found", "not_at"):
            within[clue[1][0]].append(clue)
        else:
            a, b = clue[1][0], clue[2][0]
            if a == b:
                within[a].append(clue)
            else:
                pair.setdefault((min(a, b), max(a, b)), []).append(clue)

    all_perms = list(itertools.permutations(range(n)))

    def within_ok(a, perm):
        return all(clue_holds(c, {a: perm}) for c in within[a])

    def pair_ok(a, pa, b, pb):
        pos = {a: pa, b: pb}
        return all(clue_holds(c, pos) for c in pair.get((min(a, b), max(a, b)), []))

    feasible = [[p for p in all_perms if within_ok(a, p)] for a in range(m)]
    for a in range(m):
        if not feasible[a]:
            return 0

    solutions = 0
    nodes = 0

    def rec(assigned):
        nonlocal solutions, nodes
        nodes += 1
        if nodes > node_budget:
            raise BudgetExceeded
        if solutions >= max_solutions:
            return
        if len(assigned) == m:
            solutions += 1
            return
        unassigned = [a for a in range(m) if a not in assigned]
        a = min(unassigned, key=lambda b: len(feasible[b]))
        snapshot = {b: feasible[b] for b in unassigned}
        for p in snapshot[a]:
            assigned[a] = p
            ok = True
            for b in unassigned:
                if b == a:
                    continue
                feasible[b] = [q for q in snapshot[b] if pair_ok(a, p, b, q)]
                if not feasible[b]:
                    ok = False
                    break
            if ok:
                rec(assigned)
            for b in unassigned:
                feasible[b] = snapshot[b]
            del assigned[a]
            if solutions >= max_solutions:
                return

    rec({})
    return solutions


def candidate_clues(n, m, pos):
    """All clues true in solution pos (canonical, deduplicated)."""
    seen = set()
    clues = []

    def add(clue):
        key = tuple(clue)
        if key not in seen:
            seen.add(key)
            clues.append(clue)

    for a in range(m):
        for v in range(n):
            h = pos[a][v]
            add(("found", (a, v), h))
            for h2 in range(n):
                if h2 != h:
                    add(("not_at", (a, v), h2))
    for a in range(m):
        for b in range(a, m):
            for v1 in range(n):
                for v2 in range(n):
                    if a == b and v1 == v2:
                        continue
                    x, y = (a, v1), (b, v2)
                    p1, p2 = pos[a][v1], pos[b][v2]
                    if a != b:
                        if p1 == p2:
                            add(("same", x, y))
                        else:
                            add(("not_same", x, y))
                    if p1 + 1 == p2:
                        add(("dleft", x, y))
                    if p2 + 1 == p1:
                        add(("dleft", y, x))
                    if p1 < p2:
                        add(("left", x, y))
                    if p2 < p1:
                        add(("left", y, x))
                    if abs(p1 - p2) == 1:
                        lo, hi = min(x, y), max(x, y)
                        add(("adj", lo, hi))
                    if abs(p1 - p2) == 2:
                        lo, hi = min(x, y), max(x, y)
                        add(("gap2", lo, hi))
                    if abs(p1 - p2) == 3:
                        lo, hi = min(x, y), max(x, y)
                        add(("gap3", lo, hi))
    return clues


def make_zebra(n_houses, m_attrs, rng, solver_calls_budget=400):
    """Generate one puzzle: solution grid, minimal unique clue set, text."""
    chosen = [0] + rng.sample(range(1, len(ATTRS)), m_attrs - 1)  # name always first
    values = []
    for a in chosen:
        pool = ATTRS[a][2][:]
        rng.shuffle(pool)
        values.append(pool[:n_houses])
    pos = {}
    for a in range(m_attrs):
        pos[a] = tuple(rng.sample(range(n_houses), n_houses))

    pool = candidate_clues(n_houses, m_attrs, pos)
    rng.shuffle(pool)
    working = pool[:3 * n_houses * m_attrs + 6]
    rest = pool[len(working):]

    # Grow until uniquely solvable.
    while True:
        try:
            count = count_solutions(n_houses, m_attrs, working, max_solutions=2)
        except BudgetExceeded:
            count = -1
        if count == 1:
            break
        if not rest:
            return None
        working.extend(rest[:8])
        rest = rest[8:]

    # Minimize: drop clues whose removal keeps the solution unique.
    calls = 0
    for _ in range(3):
        order = working[:]
        rng.shuffle(order)
        removed_any = False
        for clue in order:
            if calls >= solver_calls_budget:
                break
            trial = [c for c in working if c != clue]
            calls += 1
            try:
                if count_solutions(n_houses, m_attrs, trial, max_solutions=2) == 1:
                    working = trial
                    removed_any = True
            except BudgetExceeded:
                continue
        if calls >= solver_calls_budget or not removed_any:
            break

    # Final uniqueness proof with a generous budget; give up the puzzle if
    # even that cannot prove it (should not happen at these sizes).
    try:
        if count_solutions(n_houses, m_attrs, working, max_solutions=2,
                           node_budget=2000000) != 1:
            return None
    except BudgetExceeded:
        return None

    clue_lines = [render_clue(c, chosen, values) for c in working]
    rng.shuffle(clue_lines)
    background = [f"- {ATTRS[chosen[a]][0]}: " + ", ".join(values[a]) for a in range(m_attrs)]
    header = ["House"] + [ATTRS[chosen[a]][0] for a in range(m_attrs)]
    rows = []
    for h in range(n_houses):
        row = [str(h + 1)]
        for a in range(m_attrs):
            row.append(values[a][pos[a].index(h)])
        rows.append(row)
    problem = (
        f"There are {n_houses} houses, numbered 1 to {n_houses} from left to right. "
        "Each house is occupied by a different person. Each person has exactly one "
        "value for each of the following characteristics, and no two people share "
        "the same value for a characteristic:\n"
        + "\n".join(background)
        + "\n\nClues:\n"
        + "\n".join(f"{i + 1}. {line}" for i, line in enumerate(clue_lines))
    )
    return {
        "answer": {"header": header, "rows": rows},
        "problem": problem,
        "clue_count": len(clue_lines),
    }


def generate_zebra(spec, seed):
    puzzles = []
    rng = random.Random(seed)
    counter = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        size, _, count = part.partition(":")
        n, _, m = size.lower().partition("x")
        n, m, count = int(n), int(m), int(count)
        if not (2 <= n <= 6 and 2 <= m <= 6):
            raise SystemExit(f"error: zebra size {size} out of supported range 2x2..6x6")
        key = f"{n}x{m}"
        counter.setdefault(key, 0)
        for _ in range(count):
            made = None
            for _attempt in range(4):
                made = make_zebra(n, m, rng)
                if made:
                    break
            if not made:
                raise SystemExit(f"error: could not generate a unique {key} puzzle")
            counter[key] += 1
            puzzles.append({
                "id": f"zebra-{key}-{counter[key]:02d}",
                "family": "zebra",
                "difficulty": "frontier",
                "answer_kind": "grid",
                "size": key,
                "clue_count": made["clue_count"],
                "problem": made["problem"],
                "answer": made["answer"],
            })
    return puzzles


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def answer_sha256(answer):
    canonical = json.dumps(answer, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(args, problems, aime_revision, aime_rows):
    return {
        "seed": args.seed,
        "aime_count": args.aime_count,
        "zebra_spec": args.zebra_spec,
        "sources": {
            "aime": {
                "dataset": AIME_DATASET,
                "revision": aime_revision,
                "rows_available": aime_rows,
            },
            "zebra": {
                "generator": "bench/reasoning/fetch_frontier.py",
                "construction": "random solution grid -> all true clues -> "
                                "uniqueness-verified minimization (CSP solver)",
            },
        },
        "entries": [{"id": p["id"], "answer_sha256": answer_sha256(p["answer"])} for p in problems],
    }


def check_against_manifest(out_path):
    if not out_path.is_file():
        raise SystemExit(f"error: no cached problem file at {out_path}; run without --check first")
    if not MANIFEST_PATH.is_file():
        raise SystemExit(f"error: no manifest at {MANIFEST_PATH}; run without --check first")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    problems = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {e["id"]: e["answer_sha256"] for e in manifest["entries"]}
    seen = [p["id"] for p in problems]
    if seen != list(expected):
        print(f"MISMATCH: problem ids differ from manifest order/content")
        return 1
    bad = [p["id"] for p in problems if answer_sha256(p["answer"]) != expected[p["id"]]]
    if bad:
        print(f"MISMATCH: answers changed upstream for: {', '.join(bad)}")
        print("Refetch (same seed/spec) and re-freeze the manifest; results are no longer comparable.")
        return 1
    print(f"OK: {len(problems)} problems match the manifest "
          f"(seed {manifest['seed']}, aime {manifest['aime_count']}, zebra '{manifest['zebra_spec']}')")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aime-count", type=int, default=15,
                    help="AIME 2026 problems to take, in problem_idx order (0 disables; max 30)")
    ap.add_argument("--zebra-spec", default="3x4:2,4x4:5,4x5:3,5x5:3,6x4:2,6x6:1",
                    help='zebra puzzles as "NxM:count,..."; N houses x M characteristics')
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check", action="store_true",
                    help="verify the cached frontier.jsonl still matches the manifest, then exit")
    args = ap.parse_args()

    out_path = Path(args.out)
    if args.check:
        return check_against_manifest(out_path)

    problems = []
    aime_revision, aime_rows = None, 0
    if args.aime_count:
        print(f"fetching {args.aime_count} AIME 2026 problems from {AIME_DATASET} ...", flush=True)
        aime, aime_revision, aime_rows = fetch_aime(args.aime_count)
        problems.extend(aime)
        print(f"  got {len(aime)} problems (dataset revision {aime_revision[:12]})", flush=True)
    if args.zebra_spec:
        print(f"generating zebra puzzles: {args.zebra_spec} (seed {args.seed}) ...", flush=True)
        started = time.perf_counter()
        zebra = generate_zebra(args.zebra_spec, args.seed)
        problems.extend(zebra)
        clue_counts = [p["clue_count"] for p in zebra]
        print(f"  generated {len(zebra)} puzzles in {time.perf_counter() - started:.1f}s "
              f"(clues: min {min(clue_counts)}, max {max(clue_counts)})", flush=True)

    if not problems:
        print("nothing selected (aime-count 0 and empty zebra-spec)", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    manifest = build_manifest(args, problems, aime_revision, aime_rows)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(problems)} problems)")
    print(f"wrote {MANIFEST_PATH} ({len(manifest['entries'])} pinned entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
