# looped-latent — a looped mid-block retrofit of Qwen3-0.6B (v0)

**The owner's idea.** "A looped transformer … the model does its reasoning in multiple forward passes using the
fuller-precision hidden states, then outputs a token after the additional latent reasoning." For each token, a
middle block of layers runs K times on the continuous (bf16) residual stream instead of reasoning in written tokens,
then the rest of the stack emits one token. Background, methods table and sources: `RESEARCH.md` (§4 is this pack).

**Question at this scale.** Given the same tiny LoRA budget and the same wall-clock training time, does an
identity-initialized mid-block loop (K > 1) beat the same model fine-tuned without a loop, on held-out problems that
need serial computation per token and are answered with no written reasoning?

**Model.** Qwen3-0.6B (cached, pinned snapshot `c1899de2`, bf16, 28 layers, d = 1024), the post-trained
hybrid-think checkpoint used with thinking disabled (empty `<think></think>`, answer only).

**Task and data (frozen generator, `frozen/run/progen.py`, no LLM, no licence issues).** Short straight-line Python
programs over 3 variables: arithmetic mod 100, swaps, small `if/else`, `for _ in range(3)` blocks, ending in
`print(v)`. The prompt asks for the printed value; the target is just the number and `<|im_end|>` (≤ 6 generated
tokens, greedy, so there is no room for written reasoning). Answers come from executing the program. Splits, each
from its own generator seed, hash-deduplicated across splits: train 100,000 (2–8 steps); validation 2,000 and holdout
2,000 (2–8 steps); a hard guard set of 1,000 per split (9–12 steps, never trained on). The difficulty knobs are the
`DIFFICULTY` constant in `progen.py`; they are tuned in the GPU pilot so B1 lands at 40–70 % on validation.

**Arms.** *B1 = the baseline surface* (no loop, K = 1): LoRA r16 on every attention and MLP projection of all 28
layers, AdamW, cosine schedule, the harness's fixed wall-clock budget and data stream. The loop arm is the same
surface with `LOOPS_TRAIN`/`LOOPS_EVAL` > 1: layers 12–15 rerun as
`h ← h + gate_j ⊙ (block(h + W_inj·p) − h)` with zero-initialized gates and injection, so any K equals the base
model at init. Because the budget is wall-clock, B1 sees more examples than a loop arm (compute-matched, as in
RESEARCH.md). B0 (base model zero-shot, `--train-seconds 0`) is a pilot reference only: the untrained chat model
explains instead of answering, so it scores ~0 under the answer-only format.

**Metric.** Exact-match accuracy of the greedy answer on the 2–8-step set (validation in the loop, holdout in
FINALIZE), one 0/1 item per problem (item pack). The evaluator decodes the generated token ids itself and compares
the stripped text to the gold number; hedges, words or punctuation score 0.

**Budget (calibrate in the pilot, then fix).** `--train-seconds 660` in `pack.yaml` (≲ 11 min on an idle GB10,
including `build`), `budget: {unit: train_seconds, limit: 720}`; decoding 3,000 items plus the guards should take
≲ 3 min (full forward passes, no KV cache).

**MES (fixed before calibration): 0.03** (3 points; RESEARCH.md §4). Holdout SE = √(0.2/2000 + 2σ²/3): powered
(2·SE ≤ 0.03) iff the calibration σ ≤ 0.0137; at σ = 0.01, SE = 0.013. If σ is larger, enlarge `N_EVAL` under a new
tag, never the MES. Seeds: calibration [1–5], screen 1, confirm [2, 3], holdout [101–103].

**Guards.** `train_s ≤ 1.3×` baseline; effective depth `depth_ratio` (decoder-layer calls counted by frozen hooks /
28) in [1.05, 2.0], so every keep really loops at inference and costs ≤ 2× the base; OASST2 retention `text_nll`
(256 English OASST2 messages × 256 tokens from the cached snapshot, per-token NLL computed by the harness from raw
logits) ≤ 1.02× the no-loop arm (the ratio to the untouched base model is reported as `text_nll_vs_base`);
`hard_acc` ≥ 0.88× baseline (≈ 2·SE of noise at p ≈ 0.3; recalibrate after the pilot); `trainable_m ≤ 18`
(~3 % of the base); `peak_mem_gb ≤ 40`. Divergence (non-finite loss or logits) and non-causal models are `invalid`.

**Honest caveat.** Every published retrofit that gained used ≥ 1 B tokens (McLeish 26–50 B, ETD 50 B, LoopUS 3 B).
One run here sees ~5 M tokens, **roughly 1000× less**; nothing in the literature predicts what happens at this scale.
Prior that v0 clears +3 points over B1: ~30–40 % (RESEARCH.md §5). A win on a synthetic tracing task says little about
agentic coding; the next steps after a pass are Qwen3-1.7B, then CRUXEval-O as the gate.

## Open risks (for REVIEW.md)
1. **Non-loop gains can ride along.** The agent may also change LR, LoRA rank or schedule; a keep then mixes loop and
   tuning gains. The `depth_ratio ≥ 1.05` guard only ensures a keep loops. Before claiming a loop effect, rerun the
   final incumbent with `LOOPS_TRAIN = (1,)`, `LOOPS_EVAL = 1` as a `references` entry under a new tag.
2. **Wall-clock budget on a shared GPU.** Throughput changes (thermal state, owner's jobs without a GPU process,
   unified-memory pressure) change the number of steps; σ from 5 calibration seeds absorbs part of it, and contended
   runs are re-run, but slow-but-uncontended periods favour whichever arm ran then.
3. **In-process surface.** The surface runs inside the harness process (as in every arlab training pack): it could
   in principle patch the timer, the hooks or `torch` itself, or read the unlabelled eval prompts mounted at
   `/data/public` for test-time training. No answers are reachable from RUN (private data is never mounted there).
   Good-faith proposer; `program.md` forbids it; diffs are in the report.
   `depth_ratio` counts calls to the original decoder-layer modules only; a surface that recreates layers would
   under-count (the `train_s` guard still bounds total time).
4. **Hard-guard and text-guard thresholds** (0.88×, 1.02×) are set from assumed noise, not measured noise.
5. **Format and checkpoint.** The cached checkpoint is post-trained; B0 is ~0 because of format, not ability. If the
   loop is flat, RESEARCH.md suggests trying Qwen3-0.6B-Base (a download, new tag).
6. **Not built from RESEARCH.md §4:** B2 (pause tokens), the accuracy-vs-K sweep, CRUXEval-O/GSM8K transfer, and the
   Ouro/Huginn "does the task need depth" check. Run them in the pilot or a follow-up pack if v0 shows signal.
