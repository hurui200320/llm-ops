#!/usr/bin/env python3
"""Write or verify the stamps Maya's installer keeps next to a compiled engine.

maya.py only reuses a prebuilt engine when build-hip/MAYA-BUILD.json matches the source it is run from
(`build_step`: the same source hash, the same architectures, the same resolved ROCm root and the target in its
list), and only reuses the vision encoder when build-vision/MAYA-BUILD.json does (`vision_step`: the tools/vision
source hash and the architectures).  The image compiles both engines itself, so this script writes exactly the
stamp `compile_engine_hip` / `compile_vision` would have written, using maya's own `setup.source_hash` (maya.py
imports setup.py the same way; nothing runs on import).  `verify` re-checks an image without a GPU: the same
computations the installer will run on first start.

Usage: stamps.py write|verify --root /opt/maya --arch gfx1100 --engines maya,maya-ds4 [--vision] [--rocm /opt/rocm]
"""

import argparse
import json
import sys
import time
from pathlib import Path


def describe(a) -> dict:
    root = Path(a.root).resolve()
    rocm = Path(a.rocm).resolve()
    archs = [x for x in a.arch.split(",") if x]
    engines = [x for x in a.engines.split(",") if x]
    return {
        "root": root,
        "rocm": rocm,
        "archs": archs,
        "engines": engines,
        "llama": root / "third_party" / "llama.cpp",
        "engine_stamp": root / "build-hip" / "MAYA-BUILD.json",
        "vision_stamp": root / "build-vision" / "MAYA-BUILD.json",
        "engine_exes": [root / "build-hip" / t for t in engines],
        "vision_exe": root / "build-vision" / "bin" / "maya-vision",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("write", "verify"))
    ap.add_argument("--root", default="/opt/maya", help="the Maya tree the installer runs from")
    ap.add_argument("--arch", default="gfx1100", help="HIP architectures, comma-separated")
    ap.add_argument("--engines", default="maya,maya-ds4", help="the engine targets that were built")
    ap.add_argument("--vision", action="store_true", help="the GPU vision encoder was built")
    ap.add_argument("--rocm", default="/opt/rocm", help="the ROCm root the engines were built with")
    a = ap.parse_args()

    d = describe(a)
    sys.path.insert(0, str(d["root"]))
    import setup as S  # noqa: E402  (nothing runs on import)

    engine_src = S.source_hash(S.ENGINE_SOURCES)
    vision_src = S.source_hash(("tools/vision",))
    today = time.strftime("%Y-%m-%d %H:%M")

    if a.mode == "write":
        # the shape compile_engine_hip's meta has (backend, src, archs, rocm, gpu_ids, llama, lib_dirs, date, targets)
        stamp = {
            "backend": "hip",
            "src": engine_src,
            "archs": d["archs"],
            "rocm": str(d["rocm"]),
            "gpu_ids": [0],
            "llama": str(d["llama"]),
            "lib_dirs": [str(d["rocm"] / "lib")],
            "date": today,
            "targets": d["engines"],
        }
        d["engine_stamp"].write_text(json.dumps(stamp, indent=1), encoding="utf-8")
        print(f"{d['engine_stamp']}: {json.dumps(stamp)}")
        if a.vision:
            # the shape compile_vision writes (src, archs, llama, backend, date)
            vstamp = {"src": vision_src, "archs": d["archs"], "llama": str(d["llama"]), "backend": "hip",
                      "date": today}
            d["vision_stamp"].write_text(json.dumps(vstamp, indent=1), encoding="utf-8")
            print(f"{d['vision_stamp']}: {json.dumps(vstamp)}")
        return 0

    # verify: the installer's own conditions (build_step's hip branch and vision_step), plus the binaries
    problems = []
    stamp = json.loads(d["engine_stamp"].read_text(encoding="utf-8"))
    if stamp.get("backend") != "hip":
        problems.append(f"engine stamp backend is {stamp.get('backend')!r}, not 'hip'")
    if stamp.get("src") != engine_src:
        problems.append(f"engine stamp src {stamp.get('src')!r} != source_hash {engine_src!r} (a rebuild would start)")
    if stamp.get("archs") != d["archs"]:
        problems.append(f"engine stamp archs {stamp.get('archs')!r} != {d['archs']!r}")
    if stamp.get("rocm") != str(d["rocm"]):
        problems.append(f"engine stamp rocm {stamp.get('rocm')!r} != resolved {str(d['rocm'])!r}")
    missing_targets = [t for t in d["engines"] if t not in (stamp.get("targets") or [])]
    if missing_targets:
        problems.append(f"engine stamp targets {stamp.get('targets')!r} miss {missing_targets}")
    for exe in d["engine_exes"]:
        if not exe.is_file():
            problems.append(f"{exe} is missing")
    if a.vision:
        vstamp = json.loads(d["vision_stamp"].read_text(encoding="utf-8"))
        if vstamp.get("src") != vision_src:
            problems.append(f"vision stamp src {vstamp.get('src')!r} != source_hash {vision_src!r}")
        if not set(d["archs"]) <= set(vstamp.get("archs") or []):
            problems.append(f"vision stamp archs {vstamp.get('archs')!r} do not cover {d['archs']!r}")
        if not d["vision_exe"].is_file():
            problems.append(f"{d['vision_exe']} is missing")
    if problems:
        for p in problems:
            print(f"stamp check FAILED: {p}", file=sys.stderr)
        return 1
    print(f"stamps OK: engines {d['engines']} src={engine_src} archs={d['archs']} rocm={d['rocm']}"
          + (f"; vision src={vision_src}" if a.vision else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
