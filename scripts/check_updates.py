#!/usr/bin/env python3
"""
check_updates.py — Check local GGUF weights for upstream updates

Fetch SHA-256 digests from Hugging Face LFS pointers (only a few hundred bytes)
and compare them with local *.gguf.sha256 files without downloading the weights.

Each GGUF file has a .sha256 companion file alongside it. Its path relative to
the model root, without the .sha256 suffix, matches the Hugging Face repository
path and is used directly for remote lookup without searching directories.

Usage:
    python check_updates.py              # Check all models
    python check_updates.py -m 26B       # Check only model paths containing "26B"

Exit codes: 0 = no updates or warnings; 1 = updates, warnings, or no matching models
"""

import re
import sys
import argparse
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse

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

# ---------------------------------------------------------------------------
# YAML front matter parsing
# ---------------------------------------------------------------------------

def parse_front_matter(readme_path: Path) -> dict | None:
    """
    Parse YAML-style front matter (a --- block) at the start of a Markdown file.

    Returns:
      - dict: Parsed fields; may be empty if there are no valid key-value pairs.
      - None: The file does not start with ---, so no front matter was found.
    """
    with open(readme_path, encoding="utf-8") as f:
        first_line = f.readline().rstrip("\n")
        if first_line != "---":
            return None
        fields = {}
        for line in f:
            line = line.rstrip("\n")
            if line == "---":
                break
            stripped = line.lstrip()
            if stripped.startswith("#") or not stripped:
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                fields[key.strip()] = value.strip()
    return fields

# ---------------------------------------------------------------------------
# Source URL parsing
# ---------------------------------------------------------------------------

def parse_hf_repo(source_url: str) -> tuple[str, None] | tuple[None, str]:
    """
    Parse the source URL, validate the Hugging Face host, and extract the repo ID.

    Return (repo_id, None) or (None, error_description).
    """
    if not source_url:
        return None, "The source field is empty"

    parsed = urlparse(source_url)

    if parsed.scheme not in ("http", "https"):
        return None, f"Unsupported URL scheme: {parsed.scheme!r}"

    if parsed.netloc != "huggingface.co":
        return None, f"Unsupported source host: {parsed.netloc!r} (currently only huggingface.co is supported)"

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None, f"Cannot parse owner/repo from URL path: {parsed.path!r}"

    return f"{parts[0]}/{parts[1]}", None

# ---------------------------------------------------------------------------
# Hugging Face LFS pointer fetching
# ---------------------------------------------------------------------------

def fetch_lfs_sha256(
    repo_id: str, remote_path: str
) -> tuple[str, int | None] | tuple[None, str]:
    """
    Fetch a Hugging Face LFS pointer and extract its SHA-256 digest.

    remote_path is the path within the repository, matching the local companion
    file's path relative to the model directory without the .sha256 suffix,
    e.g. "BF16/file.gguf" or "file.gguf".

    Returns:
      (sha256_str, size_int)    On success (size may be None).
      (None, error_description) On failure.
    """
    url = f"https://huggingface.co/{repo_id}/raw/main/{remote_path}"
    req = Request(url, headers={"User-Agent": "check_updates/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        if e.code == 404:
            return None, "Remote file not found (404)"
        return None, f"HTTP error {e.code}"
    except URLError as e:
        return None, f"Network error: {e.reason}"

    sha_match = re.search(r"oid sha256:([a-f0-9]{64})", content)
    if not sha_match:
        return None, "Response is not an LFS pointer (the file may be small and stored inline; comparison is not currently supported)"

    size_match = re.search(r"size (\d+)", content)
    size = int(size_match.group(1)) if size_match else None
    return sha_match.group(1), size

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

# ---------------------------------------------------------------------------
# Per-model update checks
# ---------------------------------------------------------------------------

def check_model(model_dir: Path) -> tuple[int, int, int]:
    """
    Check a single model directory.

    Return (updates, warns, total):
      updates — Number of files with upstream changes.
      warns   — Number of model-level or file-level warnings.
      total   — Total number of files checked.
    """
    rel_path    = model_dir.relative_to(LIBRARY_ROOT)
    readme_path = model_dir / "README.md"

    # --- Check that the README exists ---
    if not readme_path.exists():
        print(f"\n{YELLOW}WARN{RESET}  {rel_path}")
        print(f"       README.md not found, skipping")
        return 0, 1, 0

    # --- Check front matter ---
    front_matter = parse_front_matter(readme_path)
    if front_matter is None:
        print(f"\n{YELLOW}WARN{RESET}  {rel_path}")
        print(f"       No YAML front matter (--- block) found in README.md, skipping")
        print(f"       Add source: <URL> in front matter at the start of README.md to enable update checks")
        return 0, 1, 0

    source_url = front_matter.get("source", "")

    # --- Parse the source URL ---
    repo_id, err = parse_hf_repo(source_url)
    if repo_id is None:
        print(f"\n{YELLOW}SKIP{RESET}  {rel_path}")
        print(f"       {err}")
        return 0, 0, 0

    # --- Discover all .gguf.sha256 files ---
    sha256_files = sorted(model_dir.glob("**/*.gguf.sha256"))
    if not sha256_files:
        print(f"\n{YELLOW}WARN{RESET}  {rel_path}")
        print(f"       No .gguf.sha256 files found, skipping")
        return 0, 1, 0

    print(f"\n{BOLD}{CYAN}{rel_path}{RESET}  {DIM}[{repo_id}]{RESET}")

    updates = warns = 0
    for sha256_file in sha256_files:
        # Derive the remote HF path from the companion path relative to the model root.
        remote_path = sha256_file.relative_to(model_dir).as_posix().removesuffix(".sha256")
        local_hash  = sha256_file.read_text().strip()

        result = fetch_lfs_sha256(repo_id, remote_path)

        # Show the full relative path for nested files, or just the name for root files.
        label = remote_path

        if result[0] is None:
            _, err_msg = result
            print(f"  {YELLOW}WARN  {label}{RESET}")
            print(f"        {err_msg}")
            warns += 1
        else:
            remote_hash, size = result
            if remote_hash != local_hash:
                size_hint = f"  ({fmt_size(size)})" if size is not None else ""
                print(f"  {RED}{BOLD}UPDATE{RESET}  {label}{size_hint}")
                print(f"          local:  {local_hash}")
                print(f"          remote: {remote_hash}")
                updates += 1
            else:
                print(f"  {GREEN}OK{RESET}    {label}")

    return updates, warns, len(sha256_files)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check local GGUF weights for upstream updates on Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python check_updates.py\n"
            "  python check_updates.py -m 26B\n"
            "  python check_updates.py -m gemma-4/google-E4B-it\n"
        ),
    )
    parser.add_argument(
        "--model", "-m",
        metavar="PATTERN",
        help="Only check model directories whose paths contain this string (case-insensitive)",
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

    total_updates = total_warns = total_files = 0
    for model_dir in all_dirs:
        u, w, f = check_model(model_dir)
        total_updates += u
        total_warns   += w
        total_files   += f

    # Summary
    print(f"\n{BOLD}{'─' * 52}{RESET}")
    if total_updates == 0 and total_warns == 0:
        print(f"{GREEN}All {total_files} files are up to date.{RESET}")
        sys.exit(0)
    else:
        parts = []
        if total_updates:
            parts.append(f"{RED}Files with updates: {total_updates}{RESET}")
        if total_warns:
            parts.append(f"{YELLOW}Warnings: {total_warns}{RESET}")
        print("  ".join(parts) + f"  {DIM}(Total files checked: {total_files}){RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
