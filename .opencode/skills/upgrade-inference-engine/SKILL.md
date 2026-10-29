---
name: Upgrade Inference Engine
description: Standard operating procedure for upgrading inference engines (such as llama.cpp and Strata). Use whenever checking for updates, evaluating new releases/commits, or upgrading inference engine images and version pins.
---

# Skill: upgrade-inference-engine

Do not blindly bump version numbers or pull the latest container images when upgrading an inference engine. Always inspect the upstream git diff first, evaluate how the changes affect our deployment, and present the findings before performing the upgrade.

## Workflow

### 1. Discover Current Usage & Constraints
Inspect the repository (`AGENTS.md`, `deploy/`, `.github/workflows/`, and relevant `notes/`) to identify:
- How the target engine is currently built, pulled, or wrapped (including any local entrypoint forks, build scripts, or workflow patches).
- Which CLI flags, environment variables, engine settings, and model architectures are actively used.
- Any documented workarounds, hardware constraints, and optimization guidelines.

### 2. Resolve Version Range (`<old_ref>..<new_ref>`)
1. Load the `git-reference` skill to locate (or clone) the engine's upstream repository under `~/git` and fetch the latest refs/tags.
2. Identify `<old_ref>`: the exact commit or tag currently deployed (from pinned configs/workflows, notes, or the local container image).
3. Identify `<new_ref>`: the target commit, release tag, or candidate image revision.
   - If an engine distributes pre-built binaries/packages separately from source tags, check whether the binary package itself changed between `<old_ref>` and `<new_ref>`.

### 3. Analyze the Upstream Diff
Inspect `git log` and `git diff` across `<old_ref>..<new_ref>` and evaluate four questions against our usage:

1. **Breaking changes on CLI, parameters, or integration**:
   - Did any CLI flags, environment variables, or settings we use get removed, renamed, deprecated, or change their default behavior?
   - Did upstream changes break or conflict with our local scripts, forks, or image build workflows?
2. **Bug fixes related to our usage**:
   - Are there fixes relevant to our GPU backend, model architectures, KV/checkpoint caching, speculative decoding, multimodal/vision, or tool/reasoning parsing?
   - Does the update fix an issue that allows us to retire a local patch or workaround?
3. **Performance optimizations on our usage**:
   - Are there speed, latency, loading, or memory/VRAM optimizations that apply to our hardware and models?
   - Note whether those improvements take effect automatically or require adjusting our parameters.
4. **New CLI options or knobs that may benefit us**:
   - Are there newly introduced flags or features for performance tuning, memory saving, or quality/capability improvements that fit our optimization guidelines in `AGENTS.md`?
   - Could any upstream changes make previously rejected tuning options (recorded in `notes/`) worth revisiting?

### 4. Report & Suggest Follow-Up Tests
Before modifying files or pulling/building images, present a concise upgrade report covering the four areas above:
- Propose any required config, script, or fork updates.
- If there are performance changes, parser/quality changes, or promising new knobs, suggest concrete follow-up verification or benchmark tests to the user (refer to `bench/README.md` for available test suites, or suggest tracking deferred experiments in `notes/TODO.md`).

### 5. Apply Upgrade & Verify (After User Confirmation)
Once the user approves how to proceed:
1. Update any affected configs, pins, workflows, and local fork scripts.
2. Pull or build the updated engine image and run smoke checks.
3. Run any follow-up benchmarks agreed upon with the user, append a new dated `# YYYYMMDD ...` entry to the relevant `notes/` file(s), and update `deploy/` READMEs in place so they reflect only the current state (removing any retired workarounds or superseded pins per `AGENTS.md`).
