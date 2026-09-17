# RP bench (Chinese roleplay, SillyTavern-style)

Scaffold for evaluating Chinese RP quality. Nothing here is model-specific; it
becomes useful once real scenarios exist in `scenarios/`.

## Flow

```bash
# 1. generate continuations for every scenario (server sampling defaults apply,
#    i.e. the same llama-swap config SillyTavern uses)
python3 rp_generate.py --model <llama-swap-alias>

# 2. judge with an online model (ZenMux; key stays in env, never committed)
ZENMUX_API_KEY=<your-key> python3 rp_judge.py \
    --run ../../results/rp/<alias>/ --judge-model <fixed-judge-model>
```

Output: `bench/results/rp/<alias>/<scenario>.json` (raw) + `report.json` /
`report.md` (scores). Results are gitignored; copy the report into
`notes/<family>/<variant>.md` when evaluating a model.

## Scenario format (`scenarios/*.json`)

```json
{
  "id": "short-unique-id",
  "card": {
    "name": "角色名",
    "description": "...", "personality": "...", "scenario": "...",
    "example_dialogue": "..."
  },
  "history": [{"role": "user|assistant", "content": "..."}],
  "user_message": "..."
}
```

- `system_prompt` (string, optional) replaces the prompt built from `card`
  (use it to reproduce an exact SillyTavern system prompt).
- The generator sends system + history + `user_message` and records the
  continuation. Keep histories long enough to stress memory/coherence.
- `scenarios/example.json` is a placeholder showing the format.

## Judging

Rubric lives in `rubric.md` (5 dimensions + overall, 1–5 each, Chinese).
The judge is instructed to output strict JSON; `rp_judge.py` retries once on
parse failure, then skips and records the error instead of aborting the run,
and averages per-dimension scores across scored scenarios.

**Pin the judge model.** Scores from different judges are not comparable.
Pick one strong model with good Chinese (e.g. a Claude/GLM flagship on ZenMux)
and reuse it for every model you compare.

## Rules for this public repo

- Scenarios must be SFW and free of private/copyrighted text. Do not commit
  exported SillyTavern chats containing books or private conversations;
  rewrite them.
- No API keys. `ZENMUX_API_KEY` is read from the environment only.
