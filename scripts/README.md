## Maintenance tools

Utilities for downloading GGUF model files, checking for upstream updates, and verifying local file integrity.

**Requirements:** Python 3.13 or later. All three scripts use only the standard library; no third-party packages are needed. GitHub Actions packages the scripts into the public `ghcr.io/hurui200320/llm-ops-toolbox` container image; see the [Toolbox section](../README.md#toolbox) of the root README.

Colors and progress bars are suppressed when stdout is not a TTY (for example piped output or `docker run` without `-t`), keeping cron and CI logs clean.

### Library root

Models live at `<library root>/<family>/<variant>/`, where the library root is the `LLM/` folder itself, containing the family directories:

```text
LLM/
├── gemma-4/
│   └── google-26B-A4B-it/
│       ├── README.md
│       └── BF16/
│           ├── file.gguf
│           └── file.gguf.sha256
└── ornith-1.5/
    └── ...
```

Both checkers resolve the library root independently of where the scripts are installed, with the following precedence:

1. The `--library-root PATH` argument.
2. The `LLM_LIBRARY_ROOT` environment variable.
3. The current working directory.

A root that is not an existing directory produces a clear error and exit code 2. Relative download destinations for `download_hf.py` are always resolved against the current working directory.

The toolbox image sets `LLM_LIBRARY_ROOT=/data` and installs the scripts as `check_updates`, `verify_checksums`, and `download_hf` on `PATH`. On the NAS, with the models under `/mnt/user/archive/LLM`:

```bash
# Interactive maintenance shell
docker run -it --rm --pull=always -v /mnt/user/archive/LLM:/data \
    ghcr.io/hurui200320/llm-ops-toolbox:latest

# One-shot update check (read-only mount)
docker run --rm --pull=always -v /mnt/user/archive/LLM:/data:ro \
    ghcr.io/hurui200320/llm-ops-toolbox:latest check_updates
```

Inside the interactive shell, run the commands directly from the default working directory; downloaded files and resumable `.tmp` partial downloads persist in the mounted library:

```bash
check_updates -m gemma-4
download_hf https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf /data/gemma-4/google-E2B-it/
verify_checksums -m 26B
```

Outside a container, invoke the scripts with an explicit root:

```bash
python scripts/check_updates.py --library-root /mnt/user/archive/LLM -m gemma-4
```

> **Network-mounted libraries:** over an SMB/GVfs mount (for example on the editing machine), only run `check_updates.py`. It reads a few hundred bytes of `.sha256` companions and issues small HTTP requests. The downloader writes large files and the verifier reads every byte, so both need native storage access — run them on the NAS host (through Docker) or against a local model copy on the GPU machine.

### Model README requirements

Both `check_updates.py` and `verify_checksums.py` discover files named exactly `README.md` at `<library root>/<family>/<variant>/README.md`. Each variant directory containing one is treated as a model root, and its subdirectories are recursively searched for `*.gguf.sha256` files. A variant without a README is not discovered.

Put the model README directly in the variant directory. READMEs at the library root, in family directories, or deeper inside weight subdirectories are not used for discovery. This keeps each variant's files grouped under the correct model root and remote path.

For **update checks**, the README must start with front matter containing an unquoted `source:` URL pointing to the Hugging Face repository that supplies the GGUF files. For the layout above:

```markdown
---
source: https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF
---

# Model notes

Describe the model, quantization, or local usage here.
```

- The first line must be exactly `---`, with no blank line or heading before it. Close the front matter with another `---` line.
- The parser handles simple `key: value` lines, not full YAML. Keep the URL on one line without quotes or an inline comment.
- Only `source` is used by the update checker. The Markdown body is unrestricted.
- The source must use `http` or `https` and the host `huggingface.co` or its alias `hf.co`. Use the GGUF repository, which may differ from the original model's repository.
- Update checks always use the repository's `main` revision. A branch or revision in the source URL does not change this. The update checker does not support authentication tokens.
- Missing front matter produces a warning; a missing, empty, or unsupported `source` is skipped without counting as a warning.

For **local integrity checks**, only the presence of `README.md` matters; front matter and `source` are not required. `download_hf.py` does not require or create a README, so add one yourself to make downloaded models discoverable by the checkers.

### SHA-256 file requirements

Each GGUF file to be checked needs a companion file alongside it, named by appending `.sha256` to the complete filename. For example, `BF16/file.gguf` needs `BF16/file.gguf.sha256`.

The companion file must contain **only the 64-character hexadecimal SHA-256 digest** of the GGUF file. All three scripts compare digests case-insensitively; write lowercase by convention. A trailing newline is allowed. Do not include a filename, a `sha256:` prefix, or comments. The usual `sha256sum` output containing both a digest and a filename is not accepted. A companion that is unreadable or does not contain a valid digest is reported as a warning by both checkers and never counts as an upstream update; the downloader ignores it and rewrites the companion after a verified download.

- `download_hf.py` creates or replaces the companion file after verifying the download against the upstream LFS digest.
- For files obtained another way, store the expected digest from the upstream LFS pointer or another trusted source. A digest computed only from the local file establishes a baseline for future checks; it does not confirm that the initial download matches upstream.
- Preserve each file's path relative to the Hugging Face repository. The update checker removes `.sha256` from the companion file's path relative to the model root and uses the result as the remote path. In the example layout, it checks `BF16/file.gguf` in the repository specified by `source`.
- Both checkers enumerate `*.gguf.sha256`, not `*.gguf`. A GGUF file without a companion file is ignored; a discovered model with no companions produces a warning.
- The update checker compares the stored digest with upstream without reading or checking the existence of the local GGUF file. The integrity checker requires the corresponding GGUF file and reports missing files, invalid digests, or checksum mismatches.

### check_updates.py — Check for weight updates

Fetches the SHA-256 digest from Hugging Face LFS pointer files (only a few hundred bytes) and compares it with local `*.gguf.sha256` files to detect upstream changes **without downloading the weights**.

Configure each model's README and companion files as described above. Remote files must have LFS pointers; inline files are not supported.

**Usage:**

```bash
# Check all models
python3 check_updates.py

# Check only models whose paths contain the given substring (case-insensitive)
python3 check_updates.py -m 26B
python3 check_updates.py -m gemma-4/google-E4B-it

# Check a library at an explicit root
python3 check_updates.py --library-root /mnt/user/archive/LLM
```

**Exit codes:** `0` = no updates or warnings; `1` = updates available, warnings, or no matching model directories; `2` = invalid library root. Models skipped because of a missing or unsupported source do not cause a nonzero exit, so `0` does not necessarily mean every model was checked. These codes can be used in cron jobs or automation scripts.

---

### verify_checksums.py — Verify local file integrity

For each `*.gguf.sha256` companion file, computes the SHA-256 digest of the corresponding `.gguf` file and compares it with the stored digest. Use this after downloading or when you suspect file corruption.

**Usage:**

```bash
# Verify all models
python3 verify_checksums.py

# Verify only models whose paths contain the given substring (case-insensitive)
python3 verify_checksums.py -m 26B
python3 verify_checksums.py -m gemma-4/google-E4B-it

# Verify a library at an explicit root
python3 verify_checksums.py --library-root /mnt/user/archive/LLM
```

**Exit codes:** `0` = all discovered checks passed; `1` = checksum failures, warnings, or no matching model directories; `2` = invalid library root.

| Tool | Network | Speed | Purpose |
|---|---|---|---|
| `check_updates.py` | Required | Fast (fetches only pointers) | Detect upstream changes |
| `verify_checksums.py` | Not required | Slow (reads entire files) | Verify local file integrity |
| `download_hf.py` | Required | Depends on bandwidth and local verification speed | Download new models or manually update existing ones |

---

### download_hf.py — Download model files

Downloads a single GGUF file from a Hugging Face resolve URL. It writes to a `.tmp` file in the destination directory and supports resuming interrupted downloads. Once the download finishes, it verifies the SHA-256 digest against the upstream LFS pointer, renames the temporary file to its final name, and writes the `.sha256` companion file.

Subdirectories in the URL (such as `BF16/`) are automatically recreated inside the destination directory to preserve the Hugging Face repository layout. Pass the model root as the destination, not the weight subdirectory.

If an existing companion digest matches upstream, the script verifies the local file and skips downloading when it is intact. Missing or corrupt local files are downloaded again. When the upstream digest differs from the stored companion, any leftover `.tmp` partial download belongs to the outdated version and is removed instead of resumed. File names with spaces or other special characters are percent-encoded automatically, and pasted percent-encoded URLs (`.../my%20file.gguf`) are decoded, so the local file keeps its real repository name. Paths containing `..` or a leading slash are rejected.

**URL format:** `https://huggingface.co/<owner>/<repo>/resolve/<revision>/<path>` (`hf.co` is also accepted). Downloads use the revision in the URL and require an LFS pointer for the file.

**Usage:**

```bash
# Download a file from the repository root
python3 download_hf.py \
    https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf \
    gemma-4/google-E2B-it/

# Download a file in a subdirectory (automatically creates BF16/)
# Replace file.gguf with the actual filename in the repository.
python3 download_hf.py \
    https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/BF16/file.gguf \
    gemma-4/google-26B-A4B-it/

# Gated model: supply --token or set the HF_TOKEN environment variable
python3 download_hf.py <URL> <dir> --token hf_xxxxxxxx
```

`--token` takes precedence over `HF_TOKEN`. Authentication is supported by the downloader only. Inside the toolbox image the script is installed as `download_hf` on `PATH`; use absolute destinations under `/data` there.

**Exit codes:** `0` = success; `1` = failure. On a SHA-256 mismatch, the `.tmp` file is automatically deleted.
