#!/usr/bin/env python3
"""
download_hf.py — Download GGUF files from Hugging Face

Download a file from a Hugging Face resolve URL with resume support. Write to a
.tmp file in the destination directory, then verify its SHA-256 digest. On
success, atomically rename it to the final filename and write a .sha256 companion.

Usage:
    python3 download_hf.py <URL> <target_dir> [--token TOKEN]

URL format:
    https://huggingface.co/{owner}/{repo}/resolve/{revision}/{path}

Examples:
    python3 download_hf.py \\
        https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf \\
        gemma-4/google-E2B-it/

    # File in a subdirectory (automatically creates BF16/ inside target_dir)
    python3 download_hf.py \\
        https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/BF16/file.gguf \\
        gemma-4/google-26B-A4B-it/

    # Gated model (requires a token)
    python3 download_hf.py <URL> <dir> --token hf_xxxxxxxx

The HF_TOKEN environment variable can be used instead of --token.

Exit codes: 0 = success; 1 = failure
"""

import os
import re
import sys
import time
import shutil
import hashlib
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
# Constants
# ---------------------------------------------------------------------------

ALLOWED_HOSTS = {"huggingface.co", "hf.co"}
RESOLVE_RE    = re.compile(r"^/([^/]+)/([^/]+)/resolve/([^/]+)/(.+)$")
CHUNK_SIZE    = 8 * 1024 * 1024  # 8 MB

# ---------------------------------------------------------------------------
# Terminal helper functions
# ---------------------------------------------------------------------------

def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _clear_line() -> None:
    """Blank the current line and return the cursor to its start for new output."""
    cols = shutil.get_terminal_size().columns
    sys.stdout.write(f"\r{' ' * (cols - 1)}\r")
    sys.stdout.flush()


def _print_bar(label: str, pct: float, done: int, total: int,
               speed: float | None = None) -> None:
    """Write a progress bar without a newline, overwriting it on the next call."""
    bar_width = 20
    filled    = int(bar_width * pct)
    bar       = "█" * filled + "░" * (bar_width - filled)
    pct_str   = f"{pct * 100:3.0f}%"

    size_str  = (f"  {fmt_size(done)} / {fmt_size(total)}" if total > 0
                 else f"  {fmt_size(done)}")
    speed_str = f"  {fmt_size(speed)}/s" if speed and speed > 0 else ""

    cols = shutil.get_terminal_size().columns
    suffix    = f" [{bar}] {pct_str}{size_str}{speed_str}"
    max_label = cols - len(suffix) - 3  # Two spaces of indentation plus one spare column.
    if max_label < 8:
        max_label = 8
    display = label if len(label) <= max_label else "\u2026" + label[-(max_label - 1):]

    line = f"  {display}{suffix}"
    sys.stdout.write(f"\r{line:<{cols}}")
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# URL parsing and validation
# ---------------------------------------------------------------------------

def parse_hf_url(url: str) -> tuple[str, str, str]:
    """
    Parse and validate a Hugging Face resolve URL.

    Validation rules:
      - The scheme must be http or https.
      - The hostname must be huggingface.co or hf.co.
      - The path must match /{owner}/{repo}/resolve/{revision}/{path}.

    Return (repo_id, revision, path_in_repo); print an error and exit(1) on failure.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        print(f"{RED}Error: Unsupported URL scheme {parsed.scheme!r} (http or https required){RESET}")
        sys.exit(1)

    if parsed.netloc not in ALLOWED_HOSTS:
        allowed = ", ".join(sorted(ALLOWED_HOSTS))
        print(f"{RED}Error: Unsupported hostname {parsed.netloc!r}{RESET}")
        print(f"  Supported hosts: {allowed}")
        sys.exit(1)

    m = RESOLVE_RE.match(parsed.path)
    if not m:
        print(f"{RED}Error: Unexpected URL path format{RESET}")
        print(f"  Received: {parsed.path!r}")
        print(f"  Expected: /<owner>/<repo>/resolve/<revision>/<path>")
        sys.exit(1)

    owner, repo, revision, path_in_repo = m.groups()
    return f"{owner}/{repo}", revision, path_in_repo

# ---------------------------------------------------------------------------
# LFS pointer fetching
# ---------------------------------------------------------------------------

def fetch_lfs_pointer(repo_id: str, revision: str, path_in_repo: str,
                      token: str | None) -> tuple[str, int | None]:
    """
    Fetch a Hugging Face LFS pointer and extract the expected SHA-256 and size.

    Return (sha256, size), where size may be None.
    Print an error and exit(1) on failure.
    """
    url     = f"https://huggingface.co/{repo_id}/raw/{revision}/{path_in_repo}"
    headers = {"User-Agent": "download_hf/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        detail = {
            401: "Authentication required (401); provide --token or set the HF_TOKEN environment variable",
            403: "Access denied (403); check that your token is valid and that you have accepted the model license agreement",
            404: "File not found (404); check that the URL is correct",
        }.get(e.code, f"HTTP {e.code}")
        print(f"{RED}Failed to fetch LFS pointer: {detail}{RESET}")
        sys.exit(1)
    except URLError as e:
        print(f"{RED}Failed to fetch LFS pointer: Network error: {e.reason}{RESET}")
        sys.exit(1)

    sha_match = re.search(r"oid sha256:([a-f0-9]{64})", content)
    if not sha_match:
        print(f"{RED}Failed to fetch LFS pointer: Response is not an LFS pointer{RESET}")
        print(f"  {DIM}(The file may be small and stored inline, which is not currently supported){RESET}")
        sys.exit(1)

    size_match = re.search(r"size (\d+)", content)
    size = int(size_match.group(1)) if size_match else None
    return sha_match.group(1), size

# ---------------------------------------------------------------------------
# File downloading with resume support
# ---------------------------------------------------------------------------

def download(url: str, tmp_path: Path, expected_size: int | None,
             label: str, token: str | None) -> None:
    """
    Download the URL to tmp_path with resume support and a live progress bar.
    Print an error and exit(1) on failure.
    """
    # Check whether an existing download can be resumed.
    start_byte = 0
    if tmp_path.exists():
        existing = tmp_path.stat().st_size
        if expected_size is not None and existing >= expected_size:
            print(f"  {DIM}Temporary file is already complete, skipping download{RESET}")
            return
        if existing > 0:
            start_byte = existing
            print(f"  {DIM}Resuming download from {fmt_size(existing)}{RESET}")

    headers = {"User-Agent": "download_hf/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"

    req = Request(url, headers=headers)
    try:
        resp = urlopen(req, timeout=60)
    except HTTPError as e:
        print(f"{RED}Download failed: HTTP {e.code}{RESET}")
        sys.exit(1)
    except URLError as e:
        print(f"{RED}Download failed: {e.reason}{RESET}")
        sys.exit(1)

    # Restart if the server ignores Range and returns 200 instead of 206.
    if start_byte > 0 and resp.status == 200:
        print(f"  {YELLOW}Server does not support resuming downloads, restarting from the beginning{RESET}")
        start_byte = 0

    mode = "ab" if start_byte > 0 else "wb"
    done  = start_byte
    total = expected_size or 0

    # Speed measurement window (refresh the estimate every 0.5 seconds).
    speed      = 0.0
    win_time   = time.monotonic()
    win_bytes  = done
    last_render = 0.0

    try:
        with open(tmp_path, mode) as f:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)

                now = time.monotonic()
                elapsed = now - win_time
                if elapsed >= 0.5:
                    speed    = (done - win_bytes) / elapsed
                    win_time  = now
                    win_bytes = done

                if now - last_render >= 0.1:
                    pct = done / total if total > 0 else 0.0
                    _print_bar(f"Downloading {label}", pct, done, total, speed)
                    last_render = now

        # Render final progress (100%).
        pct = done / total if total > 0 else 1.0
        _print_bar(f"Downloading {label}", pct, done, total, speed)
        sys.stdout.write("\n")
        sys.stdout.flush()

    except (IOError, OSError) as e:
        sys.stdout.write("\n")
        print(f"{RED}Write failed: {e}{RESET}")
        sys.exit(1)
    finally:
        resp.close()

# ---------------------------------------------------------------------------
# SHA-256 computation
# ---------------------------------------------------------------------------

def compute_sha256(path: Path, label: str) -> str:
    """
    Compute the file's SHA-256 digest in chunks while updating the progress bar.
    Return a 64-character hexadecimal string.
    """
    size   = path.stat().st_size
    hasher = hashlib.sha256()
    done   = 0

    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
            done += len(chunk)
            pct = done / size if size else 1.0
            _print_bar(f"Verifying   {label}", pct, done, size)

    _clear_line()
    return hasher.hexdigest()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GGUF files from Hugging Face with resume support and automatic SHA-256 verification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 download_hf.py \\\n"
            "      https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf \\\n"
            "      gemma-4/google-E2B-it/\n"
        ),
    )
    parser.add_argument("url",        help="HuggingFace resolve URL")
    parser.add_argument("target_dir", help="Local destination directory (relative or absolute path)")
    parser.add_argument(
        "--token", "-t",
        metavar="HF_TOKEN",
        help="Hugging Face API token (can also be set via the HF_TOKEN environment variable)",
    )
    args = parser.parse_args()

    # Prefer --token over the HF_TOKEN environment variable.
    token = args.token or os.environ.get("HF_TOKEN")

    # 1. Parse and validate the URL.
    repo_id, revision, path_in_repo = parse_hf_url(args.url)

    # 2. Build destination paths (target_dir / path_in_repo preserves subdirectories).
    target_dir  = Path(args.target_dir)
    final_path  = target_dir / path_in_repo
    tmp_path    = final_path.parent / (final_path.name + ".tmp")
    sha256_path = final_path.parent / (final_path.name + ".sha256")

    print(f"{BOLD}File{RESET}  {CYAN}{path_in_repo}{RESET}")
    print(f"  Source       {DIM}{repo_id}  [{revision}]{RESET}")
    print(f"  Destination  {DIM}{final_path}{RESET}")
    print()

    # Create the destination directory, including subdirectories.
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Fetch the LFS pointer (expected SHA-256 digest and file size).
    sys.stdout.write("  Fetching LFS pointer... ")
    sys.stdout.flush()
    expected_sha256, expected_size = fetch_lfs_pointer(
        repo_id, revision, path_in_repo, token
    )
    size_hint = fmt_size(expected_size) if expected_size else "Unknown size"
    print(f"{GREEN}OK{RESET}  {DIM}{size_hint}{RESET}")

    # 4. Quick local check: if stored and remote hashes match, verify the local file.
    if sha256_path.exists():
        local_hash = sha256_path.read_text(encoding="utf-8").strip()
        if local_hash == expected_sha256:
            if not final_path.exists():
                print(f"  {YELLOW}Local .sha256 matches upstream, but the .gguf file is missing, downloading again{RESET}")
            else:
                print(f"  {DIM}Local hash matches upstream, verifying local file integrity...{RESET}")
                actual = compute_sha256(final_path, path_in_repo)
                if actual == expected_sha256:
                    print(f"{GREEN}{BOLD}OK{RESET}  {final_path}")
                    print(f"  {DIM}Local file is up to date and intact, no download needed{RESET}")
                    sys.exit(0)
                else:
                    print(f"  {YELLOW}Local file is corrupt (hash mismatch), downloading again{RESET}")
                    print(f"    Expected: {expected_sha256}")
                    print(f"    Actual:   {actual}")

    # 5. Download to the temporary file.
    download(args.url, tmp_path, expected_size, path_in_repo, token)

    # 6. Verify SHA-256.
    actual_sha256 = compute_sha256(tmp_path, path_in_repo)

    if actual_sha256 != expected_sha256:
        print(f"{RED}{BOLD}SHA-256 verification failed{RESET}")
        print(f"  Expected: {expected_sha256}")
        print(f"  Actual:   {actual_sha256}")
        tmp_path.unlink()
        print(f"  {DIM}Deleted temporary file: {tmp_path.name}{RESET}")
        sys.exit(1)

    # 7. Rename the file and write its .sha256 companion.
    tmp_path.rename(final_path)
    sha256_path.write_text(actual_sha256, encoding="utf-8")

    print(f"{GREEN}{BOLD}OK{RESET}  {final_path}")
    print(f"  sha256  {DIM}{actual_sha256}{RESET}")
    print(f"  {DIM}Wrote {sha256_path.name}{RESET}")


if __name__ == "__main__":
    main()
