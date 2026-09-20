---
name: bench-tool-calling
description: Extract, analyze, and document tool-eval-bench results into the "## Tool calling" section of notes/bench.md. Use ONLY when updating tool-eval-bench results or assigning models to agentic tool-calling use cases in notes/bench.md.
---

# Skill: bench-tool-calling

Focuses exclusively on analyzing `tool-eval-bench` benchmark runs and logs, and writing the `## Tool calling` section in `notes/bench.md`.

> **STRICT BOUNDARY**:
> - This skill **MUST NOT** edit, reformat, or alter any other benchmark sections (e.g. `## Synbad`, `## Speed`, longctx, RP).
> - Only touch `notes/bench.md` under `## Tool calling`.

---

## 1. Input Artifacts

When `tool-eval-bench run` executes via `bench/run_tooleval.sh`, artifacts land in:

1. **Terminal / Execution Logs**:
   - `bench/results/tooleval/<model>-<timestamp>.log`
   - Contains final score blocks, turn durations, total token count, deployability, responsiveness, weakest categories, and safety warning banners.
2. **Markdown Reports**:
   - `bench/runs/YYYY/MM/<timestamp>_<run_id>.md`
   - Contains category breakdown tables, paired toolset deltas, per-scenario pass/fail/partial tables (`TC-01` to `TC-69`), failure error codes, and step-by-step assistant traces.
3. **Target Notes File**:
   - `notes/bench.md`, it might be a fresh file, or contains old result.

---

## 2. Extraction & Analysis Workflow

### Step 1: Extract Overall Metrics from Logs & Reports
From `bench/results/tooleval/<model>-*.log` and `bench/runs/*/*.md`, extract:
- **Score & Points**: Final score (out of 100) and earned points (e.g., `128/136 (TC-45 excluded)`).
- **Rating**: Star rating (e.g., `★★★★★ Excellent`, `★★★★ Good`).
- **Weakest Category**: Stated at the bottom of the log (e.g., `Restraint & Refusal (83%)`).
- **Safety Warning**: Check for safety warning blocks (e.g., `CRITICAL: Sleeper injection activated` or `unnecessary duplicate event`). State `None` if no warnings are triggered.
- **Duration**: Total runtime in seconds and formatted minutes (e.g., `1625.0s (about 27min)`).
- **Turn Latency & Deployability**: Median turn latency (e.g., `2.7s`), Deployability index (`α=0.7`).

### Step 2: Identify and Isolate Common Issues
Inspect scenario failures across all models under test. If an issue is a systemic trait across the family or caused by bench/harness quirks, **group it under `### Common issues`**. For example:
- **TC-45 probe timeout / budget issue**: `tool_choice='required'` probe uses 256 tokens, leading reasoning models to exceed budget before outputting a tool call.
- **Markdown code fence reflex (TC-22, TC-68)**: Models outputting ```json ... ``` instead of raw unadorned JSON string.
- **Calculator reflex (TC-11, TC-39)**: Reaching for calculator tool on trivial mental arithmetic.
- **Identity conversions (TC-35)**: Stating the same value (e.g. 500 K to K) without explaining that it's a no-op identity conversion.

Once a test case has been clasified as common issue, DO NOT repeat it in each model's individual `Failures:` list. Keep each model's failure list clean and focused on unique behaviors.

> **CRITICAL RULE**: Copy and paste the example, you should analyze the output and figure out common issues on each run.

### Step 3: Extract Model-Specific Failures & Traces
Find all scenarios marked `⚠️ partial` or `❌ fail` (excluding common issues):
- Extract the scenario ID (`TC-??`), title, and point loss (e.g. `(1/2)` or `(0/2)`).
- Inspect the prompt and assistant trace in `bench/runs/*/*.md` to understand *why* the model failed (e.g. stopped before completing turn, injected schema-violating fields, looped file searches on missing DB).
- Summarize concisely in 1–2 sentences: what happened, whether it recovered, and the real-world operational impact.

---

## 3. Formatting Structure in `notes/bench.md`

Under `## Tool calling`, maintain the following exact schema:

```markdown
### Common issues

+ TC-45 ...
+ (Other cross-model common issues)

### <model-alias>

Overview:
+ Score: <score>/100
+ Points: <points>/136 (TC-45 excluded)
+ Rating: <rating>
+ Weakest category: <category> (<percent>%)
+ Safety warning: <None or Warning Description>
+ Duration: <seconds>s (about <minutes>min)
+ Highlights:
  + (Key strengths, speed/latency, standout categories)
  + (Quantization retention or architecture advantages)
  + (Weak points summary)

Failures:

+ TC-<id>: <Title> (<points>) — <Concise explanation of failure and impact>
```

---

## 4. Writing the Tool Calling Summary Section

At the bottom of tool calling section in `notes/bench.md`, under `### Summary`, synthesize all evaluated models and map them to practical agentic use cases.

### Use Case Categories:

1. **Agentic Coding Harnesses** (e.g. OpenCode, Pi, Codex, ClaudeCode):
   - **Requirements**: Strict parameter adherence, raw JSON / structured output fidelity, resilient multi-turn instruction following, immune to malicious code snippets/repo injections, read-before-write discipline.
   - **Evaluation Criteria**: Multi-Step Chains (100%), Code Patterns (100%), Safety & Boundaries, Toolset Scale.

2. **Agentic Assistant Harnesses** (e.g. OpenClaw, Hermes Agent, daapu):
   - **Requirements**: High conversational responsiveness, graceful error recovery, polite refusal on impossible requests (no internal DB hallucinations), robust against indirect prompt injections from web/emails/external APIs.
   - **Evaluation Criteria**: Responsiveness (turn latency), Error Recovery (TC-13, TC-14), Restraint & Refusal, Sleeper Injection Resistance (TC-60).

3. **Subagents / Worker Delegation**:
   - **Requirements**: Fast turnaround, high goal completion without hanging, concise data synthesis, strict schema formatting for caller parent agent.
   - **Evaluation Criteria**: Autonomous Planning (TC-51, TC-52), 5-turn research chains (TC-62), high token efficiency.

### Summary Layout:

```markdown
### Summary

#### Model Strengths & Weaknesses Quick Reference
- `<model-alias>`: Strengths: ... | Weaknesses: ...

#### Recommended Models by Agentic Use Case
- **Agentic Coding**:
  - Best choice: ... (why: high precision, resistant to injection)
- **Agentic Assistant**:
  - Best choice: ... (why: balance of safety, error recovery, and tool accuracy)
  - Caveats: (models with critical vulnerabilities like sleeper injection must not be used on untrusted tool data)
- **Subagent / Fast Delegation**:
  - Best choice: ... (why: high throughput/low latency MoE, suitable for sandboxed internal tasks)
```
