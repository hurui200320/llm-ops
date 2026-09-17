# AGENTS.md

> **CRITICAL SECURITY NOTICE**:
> This is a PUBLIC repo, DO NOT commit confidential information.

## Quality gate

When touching anything under `scripts/` (including `scripts/test/`), run the unit tests before considering the work done (requires Python 3.13):

```bash
python3 -m unittest discover -s scripts/test
```

All tests must pass. If the change affects the toolbox build (scripts, tests, `Dockerfile`, `.dockerignore`, workflow), also build and smoke-test the image locally:

```bash
docker build -t llm-ops-toolbox:local .
docker run --rm llm-ops-toolbox:local sh -c 'for cmd in check_updates verify_checksums download_hf; do command -v "$cmd" || exit 1; done'
```

## Script policy

- Each script under `scripts/` is standalone: it must keep working when copied out of the repository on its own and use only the Python standard library.
- Do not factor shared helpers (terminal colors, `fmt_size`, library-root resolution, progress bars) into a common module; the duplication between the scripts is intentional. When changing shared behavior, apply the same change to every script carrying a copy, and extend the tests that cover both copies (e.g. `ResolveLibraryRootTest` in `scripts/test/test_check_updates.py` runs against `check_updates` and `verify_checksums`).

## Operational rule

- DO NOT touch git after making changes. User should review the change and manually stage the changes.
