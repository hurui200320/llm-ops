# Uncensored models: methods and when to use what

The canonical methods-and-decisions note for removing refusal behavior from the deployed models. Dated entries
append below (the repo's diary rule); the current decisions live in the newest entry, the deploy configs and
`notes/TODO.md`. The deciding rule for this deployment: **quality > speed > disk** - an uncensored-but-dumber
model is never acceptable, and a compliance gain that costs benchmark capability is rejected.

# 20261112 Methods, the thinking-mode boundary, and the current decisions

## The single-direction theory and Maya's runtime projection

Arditi et al. 2024 (*Refusal in Language Models Is Mediated by a Single Direction*, arXiv:2406.11717): in
instruction-tuned models the refusal behavior concentrates in a one-dimensional subspace of the residual
stream. A direction `r` (unit vector, the model's hidden size) is fitted as the difference of the mean residual
activations at the last prompt token between harmful and harmless prompt sets, at the single best layer - ranked
on held-out prompts. Maya's `tools/ablate_fit.py` does exactly this (`prompts` / `probe` / `collect` / `fit` /
`eval`); the shipped per-model files are a few KB (`direction.1` F32 + metadata).

Maya 1.1.59's `--ablate FILE` projects that direction out at runtime: every layer's attention output and
FFN/MoE output become `y <- y - s * r (r . y)` right before they join the residual stream (upstream
`docs/SETTINGS.md`, "DeepSeek: projecting a direction out" and the per-request section). The properties that
matter operationally:

- **Weights are untouched.** The pack, the quant, the model files - all unchanged. With `--ablate-default off`
  and no `ablate` field in the request, answers are bit-identical to no direction file at all.
- **Per-request switch, no restart.** `"ablate": true/false` on OpenAI Chat Completions and Responses,
  Anthropic Messages and Maya's model API; a request without the field takes the model default (the Models-page
  switch, `ablate.default`, changeable live). Our activation path is the llama-swap `:ablation` alias
  (deploy/maya/README.md).
- **Engine support** (upstream): `maya-ds4` (DeepSeek-V4.1-Flash), `maya-mm3` (MiniMax-M3.1-Flash, not with
  `--tp`), the GLM engine (GLM-5.3-Flash, GLM-5.3). **Qwen has none** - which is why the Qwen3.8-Flash-Next lane
  runs a static abliterated quant instead (below).
- **Toggle cost.** A conversation's cache was read under one setting; a request with the other setting starts it
  over (no cached prefix, restore point, kept conversation or shared start crosses the two), and an ablated
  conversation is never kept as a slot. In an agentic loop: switch on for the whole session, never per turn.
- **Cost when on**, measured on our lanes 20261112 (notes/maya.md 20261112): DS free (all pp/tg deltas inside
  the +-3 % spread), GLM pp free and tg -4.0 % (40 %) / -4.8 % (85 %) on ablated requests only.
- **Directions are model-level, not quant-level.** The engine validates structure only (a control-vector GGUF
  with one F32 `direction.N` of the hidden size). The DeepSeek direction fitted on Maya's Maya-S transferred to
  the ISTA-DASLab IQ2_XXS lane (same base model, same hidden size): refusals 99.0 % -> 3.8 % on 104 held-out
  harmful prompts, harmless false refusals 0.96 % -> 0.0 %, no coherence damage
  (notes/deepseek/v41f1017.md 20261112, think off).

**Pitfall: stock llama.cpp cannot use these files.** Maya's direction file reuses llama.cpp's control-vector
GGUF container, but llama.cpp's `--control-vector` **adds** the vector to the residual stream (additive
steering, `h += scale * d` per layer) - feed it the refusal direction at positive scale and the model is pushed
*toward* refusal. The projection is not expressible as a control vector. On stock llama.cpp the equivalent is a
**rank-1 abliteration LoRA**: for a linear layer `y = Wx`, `y - s r (r^T W x) = (W - s r r^T W) x`, i.e.
`dW = B A` with `B = -r` (dim x 1) and `A = s r^T W` (1 x dim) - a rank-1 adapter computed analytically, no
training, exportable as a GGUF LoRA (`--lora`). Maya's ds4 engine even accepts such adapters directly through
`--ablate` (upstream docs/SETTINGS.md). On big MoE models the adapter explodes in size (the edit must reach
every routed expert), which is precisely why Maya implements the projection in the engine instead.

## The thinking-mode boundary (why the DeepSeek lane dropped the ablation)

Runtime directions are fitted on thinking-off prompts, and the fit does not survive thinking. Upstream measured
it on DeepSeek-V4.1-Flash (docs/SETTINGS.md): with the refusal direction fitted at layer 22 on thinking-off
prompts, thinking "high" on 120 held-out harmful prompts - scale 1 refused **92**, scale 1.375 refused **12**
(thinking off, 256 tokens: 5 and 0). The cost is the thinking itself: ~8,800 characters median per harmful
prompt at 1.375 against ~3,400 at scale 1, ~130 s against 76 s a request - and "higher scales repeated
themselves more in the thinking; a direction refitted on thinking-on prompts did not help (one layer, one
scale)". The reading: the projection removes the refusal *trigger*, not the RLHF-trained habit of agonizing
inside the thinking block; over-projecting the whole residual stream to compensate degrades the reasoning path.

GLM measured the other way (upstream, the XS-fitted direction at 1.5): with thinking on (24 harmful prompts,
4096 tokens) it refused **none**, 4/24 ended at the token cap with no answer, against 22 refusals + 2 no-answer
for the stock model. So the runtime ablation holds under thinking on GLM and does not on DeepSeek.

**Decision (20261112):** the DeepSeek lane's ablation wiring is removed (notes/deepseek/v41f1017.md) - the lane
serves `reasoning_effort: high`, exactly where the direction does not protect; the GLM lane keeps its
`:ablation` alias. Guideline: **runtime projection for non-thinking or light usage; a static method for
thinking workloads on DeepSeek.** Never stack the projection onto weights that are already ablated (double
ablation).

## The static methods

### Abliterated weights (single direction baked in)

The Arditi projection written into the weights (`W' = W - r r^T W` on the residual writers), saved as a new
checkpoint and re-quantized for distribution. Works on any engine at zero runtime cost, cannot be turned off,
and the quality outcome depends entirely on the surgeon. On low-bit quants the edit is usually a
dequantize -> edit -> re-quantize transplant, where double rounding noise lands on top of the edit; the failure
mode of a badly done one is a model that complies but rambles, loses math/code sharpness, or loops inside its
thinking. GLM's Maya-M-Derisked quant was this category (a third-party edit, "experimental | not measured yet"
upstream) and was retired 20261112 (notes/glm/5.3-flash.md). **Worth using only when the publisher shows their
work** - which tensor classes were touched, at what precision, and quality numbers against the base quant.
Otherwise prefer the runtime projection (reversible, measured) or a verified surgical transplant (next).

### Surgical transplants on quantized bases (the SC117 / Abiray pattern)

The method that satisfies "no dumb models" for thinking workloads on big MoE. On an ISTA-DASLab GSQ-RCO quant,
the routed experts (the bulk of the parameters and most of the measured capability) stay **byte-identical** to
the base quant; only the dense residual-writer tensors are replaced with ablated weights, re-encoded at higher
precision (Q6_K/Q8_0) so the edit itself loses nothing to rounding. Reported on the model cards (not
independently verified here - re-verify sha256 and spot-check on our lanes before trusting):

- **SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated**: 1,079 of 1,223 tensors byte-identical to
  the official GSQ-RCO quant - all 512 experts' gate/up matrices, embeddings, output head and the 28.8 GB
  n-gram shard; 144 residual-writer tensors replaced, 95 of them re-encoded at Q8_0 (+0.25 GB). In practice:
  no refusals, no thinking-loop problems; tool-call boundary behavior slightly softer than vanilla
  (notes/bench.md, TC-42/TC-59) - keep vanilla Qwen for strict-boundary work.
- **Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS**: the 28 layers of
  1-bit routed experts and the 110.6 GB Engram shard untouched; 86 residual-writer tensors replaced, attention
  output upgraded to Q6_K (+0.34 GB). Reported: thinking-high refusals 0/30 (base 27/30), XSTest false refusals
  7.6 % -> 0.0 %, GPQA-Diamond 83.3 % = base, AIME25 80.0 % vs base 76.7 %, median KL 4.2e-4 nats, top-1
  agreement 98.4 % against the base quant. Shard 6 (the Engram) is reported byte-identical to the ISTA original:
  hard-link it instead of redownloading (~110 GB saved), after verifying the sha256 against the card and the
  local file.
- The **Coder variants** of these (expert-pruned, roughly half the experts) are excluded by the quality-first
  rule: pruning physically removes capability outside the calibration domain.

### Heretic (automated multi-direction search)

`p-e-w/heretic`: training-free model modification wrapped in an Optuna TPE hyperparameter search that
co-optimizes two scorers - refusal rate and KL divergence from the base model. Modifiers: **Arbitrary-Rank
Ablation (ARA)**, the default since 2.x, and the classic directional ablation with a flexible per-layer weight
kernel (projected abliteration, norm preservation/MPOA). The output is a **static** modified model (full
weights; reproducible byte-for-byte from a ~9 KB manifest - the "Grimoire" library). Independent small-model
comparisons found it the most surgical of the abliteration tools.

When it makes sense here: **a published Heretic'd quant of a model we run, with both low refusal and low KLD, is
worth a redownload** - the search optimizes exactly the trade a single fixed direction cannot (per-layer
strength). Self-running it on our models is not practical: each trial loads the model at (near-)full precision
(on-the-fly bitsandbytes 4-bit helps, but a 700B-class MoE at 4-bit is still hundreds of GB), so for big MoE we
depend on someone else's search. For small/mid dense models (the gemma-4 26B/31B class) it is the first choice
among static methods - with the standing caveat that a small model which no longer refuses may still be too
weak for complex tasks: capability, not compliance, is the binding constraint there.

### Uncensored fine-tunes (SFT/DPO)

Re-training on refusal-stripped or compliance-tilted data (Dolphin-style). The most *natural* results - it
removes hedging and the thinking-block self-censorship that linear methods cannot touch, because the change is
in the distribution, not one direction. But capability retention has the highest variance of all methods:
third-party fine-tunes of aligned bases regularly lose points on knowledge/reasoning suites (the dealignai
DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8 card, for example, self-reports MMLU 87.63 -> 83.77 against the base).
Given the quality-first rule: consider a fine-tune only when its card shows base-level scores, and prefer the
surgical transplants where they exist.

## When to use what (per model, current state)

| Model / lane | Uncensored path | Why |
| --- | --- | --- |
| GLM-5.3-Flash (`glm53f-iq2s`) | vanilla + the `:ablation` alias, on for the whole session when needed | GLM tolerates the single-direction projection well, including under thinking (0/24 refusals at 1.5 upstream); off = 100 % vanilla; tg -4..-5 % only on ablated requests. The Derisked quant was retired 20261112 (notes/glm/5.3-flash.md) |
| DeepSeek-V4.1-Flash-1017 (`dsv41f1017-iq2xxs`) | the plain ISTA quant for everyday work; the static uncensored quant (Abiray) for uncensored work - never the direction ablation | thinking workload; the think-off-fitted direction does not hold under thinking (92/120 at scale 1) and a refit does not help (notes/deepseek/v41f1017.md 20261112) |
| Qwen3.8-Flash-Next (strata, `sc117`) | the SC117 abliterated quant stays; vanilla stays for strict tool-boundary work | Maya's engine has no Qwen ablation; SC117 is the surgical-transplant pattern at its best (byte-identical experts) |

The general "when to use what", compressed:

1. **Default: no uncensoring at all.** Plain requests are bit-identical without a direction file, and the
   vanilla quant is the best-measured. Reach for an uncensored path per session or per task, not globally.
2. **Non-thinking usage on the maya engines, occasional need**: the `:ablation` alias per request/session
   (GLM lane today; the mechanism is documented in deploy/maya/README.md).
3. **Thinking usage on DeepSeek**: a static surgical transplant (the Abiray pattern) - the only method measured
   to hold under thinking without capability loss.
4. **Small/mid dense models**: a published Heretic'd quant with low KL; expect capability, not compliance, to be
   the limit.
5. **Fine-tunes**: only with card-verified base-level scores.
6. **Never combine** the runtime projection with already-ablated weights.
7. **Agentic loops**: once a session needed ablation, keep it on until the session ends - toggling restarts the
   conversation's cache (a 30-150K-token re-read) and the un-ablated model may re-refuse on the same context.

## Sources

- Upstream: project-maya `docs/SETTINGS.md` (the ablation sections), `docs/MODELS.md`, the CHANGELOG 1.1.58 and
  1.1.59 entries, `tools/ablate_fit.py`; Arditi et al. 2024 (arXiv:2406.11717); the p-e-w/heretic README.
- Our measurements: notes/maya.md 20261112 (the 1.1.59 wiring and A/B), notes/deepseek/v41f1017.md 20261112
  (the transfer eval and the removal), notes/glm/5.3-flash.md 20261112 (the derisked retirement),
  notes/bench.md (SC117's boundary behavior, the derisked tool-calling scores).
- Model-card-reported numbers (SC117, Abiray, dealignai): the repos' cards as of 2026-11. Re-verify sha256 and
  run a refusal/quality spot-check on our lanes before trusting any of them in production.
