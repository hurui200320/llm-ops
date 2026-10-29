---
name: Tune Model
description: Standard operating procedure for tuning model inference parameters (speed, VRAM, host RAM, latency, concurrency). Use whenever discovering engine knobs, planning A/B parameter experiments, or tuning a model deployment.
---

# Skill: tune-model

Do not guess or blindly toggle flags when tuning a model. Always inspect the inference engine's source code to find what knobs actually exist and how they work on the active code path, rank them for the user, test one variable at a time, and write every single test arm's result to disk before starting the next arm.

## Workflow

### 1. Discover Target, Constraints & Prior History
- **Identify the tuning target**: Default to **pp (prompt processing) and tg (token generation) speed**, unless the user specifies another objective (e.g., VRAM headroom, host RAM/swap footprint, decision latency, or multi-slot concurrency).
- **Gather project rules & current state**:
  - Read `AGENTS.md` for global optimization boundaries and non-negotiable rules (such as precision rules, context window floors, and performance targets).
  - Inspect `deploy/` (proxy configs, container macros, entrypoint wrappers) to understand how the target model and engine are currently invoked, wrapped, and constrained.
  - Read the relevant engine note (`notes/<engine>.md`) and model note (`notes/<model-family>/<model>.md`, if it exists) to see what has **already been tested, kept, or rejected**, as well as any documented engine or wrapper gotchas.

### 2. Inspect Engine Source Code & Collect Candidate Knobs
1. Load the `git-reference` skill to locate (or clone) the inference engine's repository under `~/git`, and verify the exact commit or tag matching the deployed image/binary.
2. Search the engine source code (CLI argument parsers, `getenv` lookups, kernel dispatchers, memory/cache planners, scheduler/batching logic) and local wrapper scripts in `deploy/` for all knobs relevant to the tuning target:
   - **Verify code-path applicability**: Confirm each knob actually executes on the target model architecture, quantization format, GPU backend/architecture, and multi-GPU topology (filtering out flags that are silent no-ops or belong to a different engine path).
   - **Check wrapper & launch translation**: Verify how `deploy/` scripts pass settings to the engine (e.g., container env vars vs. CLI flags, boolean/value formatting rules, or settings that force a full re-setup vs. launch-only flags).
   - **Respect quality boundaries**: Exclude knobs that alter numerical precision or degrade model quality in violation of `AGENTS.md` unless the user explicitly asks to evaluate them.
   - **Cross-check prior notes**: Skip knobs already marked as rejected in `notes/<model-family>/<model>.md`, *unless* the upstream source code on that execution path has changed since the note was recorded.

### 3. Score & Rank Candidate Knobs
Order candidate knobs from **most impactful to least impactful** so the user can decide which knobs to test and which to skip:
- **Bottleneck alignment**: Does the knob act directly on the active bottleneck for the tuning target (e.g., GPU compute kernel selection, layer/expert placement across asymmetric buses, stage balance, pipeline overlap, batch/ubatch/chunk sizing, speculative draft acceptance) rather than an idle or non-limiting subsystem?
- **Expected impact tier**:
  - **High**: Changes compute kernels, device/tier placement, stage balance, pipeline concurrency, or batching/chunk geometry.
  - **Medium**: Adjusts cache/store budgets, speculative decoding thresholds, or worker/thread pools on active paths.
  - **Low / Marginal**: Minor I/O, polling, or secondary buffer tweaks.
- **Trade-offs & risks**: Note expected side effects on other metrics or resources (e.g., trading VRAM expert capacity for larger prefill buffers, or improving `pp` at the expense of `tg` or VRAM headroom).

### 4. Plan Controlled Experiments (Control Variates) & Get Approval
Read `bench/README.md` to select the appropriate benchmark runner and workload for the model type and tuning target, and design the experiment plan:
1. **Baseline arm**: Define the exact baseline configuration and measurement command to run first under identical conditions.
2. **Control variates (one knob per arm)**: Change only **one knob** (or one tightly coupled setting) per test arm so every improvement or regression has an unambiguous cause. State clearly whether each arm builds cumulatively on the previous winning configuration or tests against the baseline (and note any suspected interaction effects worth testing).
3. **Arm hygiene & isolation**:
   - Check whether the engine or wrapper persists settings into preset/state files on disk across runs; reset or override them before each arm so settings never leak into subsequent tests.
   - Check whether repeated CLI flags override or append to earlier flags.
   - Ensure comparable cache states across arms (e.g., cold prefills when measuring `pp`, clean container shutdown, and full VRAM release between arms).
4. **Telemetry & guardrails**: Alongside the primary target metric, record secondary guardrails (both `pp` and `tg`, speculative acceptance rate, per-GPU VRAM peak/free headroom during runs, host RAM/swap/cgroup counters, and engine placement/wait log lines).

**Stop and present the ranked knob table and test plan to the user for approval before starting the tuning runs.**

### 5. Execute Tuning Loop with Incremental Note Recording
Once the user approves the plan, run the baseline and each planned test arm sequentially.

> **CRITICAL ANTI-COMPACTION RULE**:
> Tuning benchmarks run long and produce large outputs that can trigger context compaction mid-session.
> **Immediately after each arm finishes — and strictly before starting the next knob/arm — record that arm's results into `notes/<model-family>/<model>.md`** (creating the file if it does not exist). Never batch note updates at the end of the session.

For **every** tested arm (both working and non-working), record in `notes/<model-family>/<model>.md`:
- **Arm configuration**: Exact knob/value tested and the baseline it was applied onto.
- **Measured metrics & telemetry**: Target metrics (`pp`, `tg`, latency, acceptance rate) and resource envelope (free/peak VRAM per GPU, RAM/swap/cgroup behavior).
- **Verdict & mechanics**: Mark clearly as **Kept**, **Rejected**, or **Neutral**, and explain *why* using engine log facts and source-code mechanics.
- **Do-not-repeat guidance**: Summarize rejected/neutral knobs, wrapper gotchas, and what upstream condition would need to change before revisiting them in the future.

### 6. Finalize & Validate
After all approved arms are tested and recorded:
1. Validate the final kept configuration end-to-end (including syncing `deploy/` if requested, running smoke checks, and verifying clean container stop / VRAM release).
2. Keep detailed tuning tables, rejected arms, and parameter rationale in a dated `# YYYYMMDD ...` entry appended to `notes/<model-family>/<model>.md` as the single source of truth, and update `deploy/` READMEs in place to reflect only the current state per `AGENTS.md`.
