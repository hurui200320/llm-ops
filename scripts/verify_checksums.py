#!/usr/bin/env python3
"""
verify_checksums.py — Verify local GGUF file integrity using SHA-256

For each .gguf.sha256 companion file, compute the corresponding GGUF file's
SHA-256 digest and compare it with the stored hash to detect local changes.

Usage:
    python verify_checksums.py              # Verify all models
    python verify_checksums.py -m 26B       # Verify only model paths containing "26B"

Exit codes: 0 = all checks passed; 1 = failures, warnings, or no matching models
"""

import hashlib
import shutil
import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

LIBRARY_ROOT = Path(__file__).parent

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _clear_line() -> None:
    """Blank the current terminal line and return the cursor to its start."""
    cols = shutil.get_terminal_size().columns
    sys.stdout.write(f"\r{' ' * (cols - 1)}\r")
    sys.stdout.flush()


def _print_progress(label: str, pct: float) -> None:
    """Write a progress bar without a newline, overwriting it on the next call."""
    bar_width = 20
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)
    pct_str = f"{pct * 100:3.0f}%"

    cols = shutil.get_terminal_size().columns
    prefix = "  Checking "
    suffix = f" [{bar}] {pct_str}"
    max_label = cols - len(prefix) - len(suffix) - 1
    if max_label < 8:
        max_label = 8
    display = label if len(label) <= max_label else "\u2026" + label[-(max_label - 1):]

    sys.stdout.write(f"\r{prefix}{display}{suffix}")
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# SHA-256 computation
# ---------------------------------------------------------------------------

def compute_sha256(gguf_path: Path, label: str) -> str:
    """
    Read the GGUF file in chunks and compute SHA-256 while updating progress.
    Return a 64-character hexadecimal string.
    """
    file_size = gguf_path.stat().st_size
    hasher = hashlib.sha256()
    done = 0

    _print_progress(label, 0.0)
    with open(gguf_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
            done += len(chunk)
            pct = done / file_size if file_size else 1.0
            _print_progress(label, pct)

    return hasher.hexdigest()

# ---------------------------------------------------------------------------
# Per-file verification
# ---------------------------------------------------------------------------

def verify_file(sha256_file: Path, model_dir: Path) -> str:
    """
    Verify the GGUF file corresponding to a single .gguf.sha256 file.

    Return: 'ok' | 'fail' | 'missing' | 'invalid'.
    """
    gguf_path = sha256_file.with_suffix("")  # Remove the trailing .sha256 suffix.
    label = sha256_file.relative_to(model_dir).as_posix().removesuffix(".sha256")

    # --- Validate the stored hash format ---
    stored_hash = sha256_file.read_text(encoding="utf-8").strip().lower()
    if len(stored_hash) != 64 or not all(c in "0123456789abcdef" for c in stored_hash):
        _clear_line()
        print(f"  {YELLOW}INVALID{RESET}  {label}")
        print(f"           The .sha256 file does not contain a valid 64-character hexadecimal hash")
        return "invalid"

    # --- Check that the GGUF file exists ---
    if not gguf_path.exists():
        _clear_line()
        print(f"  {YELLOW}MISSING{RESET}  {label}")
        print(f"           Corresponding .gguf file not found: {gguf_path.name}")
        return "missing"

    # --- Compute and compare ---
    size_hint = fmt_size(gguf_path.stat().st_size)
    actual_hash = compute_sha256(gguf_path, label)
    _clear_line()

    if actual_hash == stored_hash:
        print(f"  {GREEN}OK{RESET}       {label}  {DIM}({size_hint}){RESET}")
        return "ok"
    else:
        print(f"  {RED}{BOLD}FAIL{RESET}     {label}  {DIM}({size_hint}){RESET}")
        print(f"           expected: {stored_hash}")
        print(f"           got:      {actual_hash}")
        return "fail"

# ---------------------------------------------------------------------------
# Per-model verification
# ---------------------------------------------------------------------------

def verify_model(model_dir: Path) -> tuple[int, int, int, int]:
    """
    Verify all GGUF files with companion checksums under a model directory.

    Return (ok, fail, warn, total).
    """
    rel_path = model_dir.relative_to(LIBRARY_ROOT)
    sha256_files = sorted(model_dir.glob("**/*.gguf.sha256"))

    if not sha256_files:
        print(f"\n{YELLOW}WARN{RESET}  {rel_path}")
        print(f"       No .gguf.sha256 files found, skipping")
        return 0, 0, 1, 0

    print(f"\n{BOLD}{CYAN}{rel_path}{RESET}")

    ok = fail = warn = 0
    for sf in sha256_files:
        result = verify_file(sf, model_dir)
        if result == "ok":
            ok += 1
        elif result == "fail":
            fail += 1
        else:
            warn += 1

    return ok, fail, warn, len(sha256_files)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify local GGUF file integrity using SHA-256.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python verify_checksums.py\n"
            "  python verify_checksums.py -m 26B\n"
            "  python verify_checksums.py -m gemma-4/google-E4B-it\n"
        ),
    )
    parser.add_argument(
        "--model", "-m",
        metavar="PATTERN",
        help="Only verify model directories whose paths contain this string (case-insensitive)",
    )
    args = parser.parse_args()

    # Discover variants at <library root>/<family>/<variant>/README.md.
    all_dirs = sorted(p.parent for p in LIBRARY_ROOT.glob("*/*/README.md")
                      if p.is_file())

    if args.model:
        pattern = args.model.lower()
        all_dirs = [d for d in all_dirs
                    if pattern in str(d.relative_to(LIBRARY_ROOT)).lower()]

    if not all_dirs:
        if args.model:
            print(f"No model directories found matching {args.model!r}")
        else:
            print("No model directories containing README.md found under <family>/<variant>/ in the library root")
        sys.exit(1)

    total_ok = total_fail = total_warn = total_files = 0
    for model_dir in all_dirs:
        ok, fail, warn, total = verify_model(model_dir)
        total_ok    += ok
        total_fail  += fail
        total_warn  += warn
        total_files += total

    # Summary
    print(f"\n{BOLD}{'─' * 52}{RESET}")
    if total_fail == 0 and total_warn == 0:
        print(f"{GREEN}All {total_files} files passed verification.{RESET}")
        sys.exit(0)
    else:
        parts = []
        if total_fail:
            parts.append(f"{RED}Files that failed verification: {total_fail}{RESET}")
        if total_warn:
            parts.append(f"{YELLOW}Warnings: {total_warn}{RESET}")
        print("  ".join(parts) + f"  {DIM}(Total files checked: {total_files}){RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
