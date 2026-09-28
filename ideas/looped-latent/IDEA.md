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

**Task and data (frozen generator, `frozen/prepare/progen.py`, no LLM, no licence issues).** Short straight-line
Python programs over 3 variables: + and − mod 100 with constants ≤ 5, swaps, small `if/else` (pilot calibration), ending in
`print(v)`. The prompt asks for the printed value; the target is just the number and `<|im_end|>` (≤ 6 generated
tokens, greedy, so there is no room for written reasoning). Answers come from executing the program. Splits, each
from its own generator seed, hash-deduplicated across splits: train 100,000 (1–6 steps); validation 2,000 and holdout
2,000 (1–6 steps); a hard guard set of 1,000 per split (7–10 steps, never trained on). The difficulty knobs are the
`DIFFICULTY` constant in `progen.py`; they are tuned in the GPU pilot so B1 lands at 40–70 % on validation (pilot 2:
B1 0.623). The generator, `prepare.py` and the seeds live in `frozen/prepare/`, mounted only into PREPARE (and the
pack tests), so RUN cannot regenerate the labels.

**Arms, and what is frozen.** Frozen for every arm (`frozen/run/looprt.py`): the base model, LoRA r16/α32 on all 7
projections of all 28 layers, the forward (prelude → loop → coda → head), the answer loss, AdamW (LR 2e-4, β
(0.9, 0.99), no WD), the warmup + cosine schedule, grad clip 1.0, batch 32, the data order and the wall-clock budget.
The surface (`surface/loop.py`) supplies only the loop module: it receives the prelude output and a frozen
`block(x)` callable (layers LOOP_START..LOOP_END−1, one pass) and returns the state handed to the coda. It never
sees token ids, layers or base weights. *B1 = the baseline surface* (no loop, K = 1: `h = block(p)`). The loop arm
reruns layers 12–15 as `h ← h + gate_j ⊙ (block(h + W_inj·p) − h)` with zero-initialized gates and injection, so any
K equals the base model at init. Because all tuning is frozen, any gain over B1 is attributable to the loop. Because the budget is wall-clock, B1 sees more examples than a loop arm (compute-matched, as in
RESEARCH.md). B0 (base model zero-shot, `--train-seconds 0`) is a pilot reference only: the untrained chat model
explains instead of answering, so it scores ~0 under the answer-only format.

**Metric.** Exact-match accuracy of the greedy answer on the 1–6-step set (validation in the loop, holdout in
FINALIZE), one 0/1 item per problem (item pack). The evaluator decodes the generated token ids itself and compares
the stripped text to the gold number; hedges, words or punctuation score 0.

**Budget (calibrate in the pilot, then fix).** `--train-seconds 660` in `pack.yaml` (≲ 11 min on an idle GB10; the
timer starts before the surface is imported), `budget: {unit: train_seconds, limit: 720}`; decoding 3,000 items, the 2,000-item loop-off
ablation and the guards should take ≲ 4 min (full forward passes, no KV cache). Pilot 2 RUN was 760–790 s before the
ablation decode and the integrity hashes were added; re-measure.

**MES (fixed before calibration): 0.03** (3 points; RESEARCH.md §4). Holdout SE = √(0.2/2000 + 2σ²/3): powered
(2·SE ≤ 0.03) iff the calibration σ ≤ 0.0137; at σ = 0.01, SE = 0.013. If σ is larger, enlarge `N_EVAL` under a new
tag, never the MES. Seeds: calibration [1–5], screen 1, confirm [2, 3], holdout [101–103].

**Guards.** `train_s ≤ 1.3×` baseline; effective depth `depth_ratio` (tokens through the loop block, metered by
the frozen `block` callable, as layer work / 28) in [1.05, 2.0], so every keep loops and costs ≤ 2× the base;
`loop_gain ≥ 0.005` (accuracy minus the same trained model decoded with the loop disabled), so the extra passes must
actually improve answers (zero gates or dummy block calls give exactly 0); OASST2 retention `text_nll`
(256 English OASST2 messages × 256 tokens from the cached snapshot, per-token NLL computed by the harness from raw
logits) ≤ 1.02× the no-loop arm (the ratio to the untouched base model is reported as `text_nll_vs_base`);
`hard_acc` ≥ 0.85× baseline (pilot 2: B1 hard 0.148, 1 − 2·0.0112/0.148); `trainable_m ≤ 18` (frozen LoRA 10.09 M +
loop parameters); `peak_mem_gb ≤ 40`. Invalid: divergence (non-finite loss or logits), non-causal models, any change to
the base weights (hash at load vs end), any parameter or buffer change after the training deadline (hash after
training vs after decoding), or a trainable set other than LoRA + the loop's parameters (checked every step).

**Honest caveat.** Every published retrofit that gained used ≥ 1 B tokens (McLeish 26–50 B, ETD 50 B, LoopUS 3 B).
One run here sees ~5 M tokens, **roughly 1000× less**; nothing in the literature predicts what happens at this scale.
Prior that v0 clears +3 points over B1: ~30–40 % (RESEARCH.md §5). A win on a synthetic tracing task says little about
agentic coding; the next steps after a pass are Qwen3-1.7B, then CRUXEval-O as the gate.

## Open risks (for REVIEW.md)
1. **Attribution (fixed in round 2).** All non-loop training settings are frozen and `loop_gain` compares every run with
   its own loop-disabled decode. Residual: a loop arm trains with different gradients through the shared LoRA (e.g.
   deep supervision), so "the loop" includes training-time effects of the loop, not only inference-time depth.
2. **Wall-clock budget on a shared GPU.** Throughput changes (thermal state, owner's jobs without a GPU process,
   unified-memory pressure) change the number of steps; σ from 5 calibration seeds absorbs part of it, and contended
   runs are re-run, but slow-but-uncontended periods favour whichever arm ran then.
3. **In-process surface.** The surface runs inside the harness process (as in every arlab training pack): it could
   in principle patch the timer, the hooks or `torch` itself, or read the unlabelled eval prompts mounted at
   `/data/public` for test-time training. No answers are reachable from RUN (private data is never mounted there).
   Good-faith proposer; `program.md` forbids it; diffs are in the report. See REVIEW.md for what is enforced.
4. **Text guard vs the pilot.** Pilot 2's K4 arm had text_nll 1.125× B1, far over the 1.02× guard: a loop that trains
   must keep general text intact, or no loop keep is possible. Thresholds (0.85×, 1.02×) are from assumed noise.
5. **Format and checkpoint.** The cached checkpoint is post-trained; B0 is ~0 because of format, not ability. If the
   loop is flat, RESEARCH.md suggests trying Qwen3-0.6B-Base (a download, new tag).
6. **Not built from RESEARCH.md §4:** B2 (pause tokens), the accuracy-vs-K sweep, CRUXEval-O/GSM8K transfer, and the
   Ouro/Huginn "does the task need depth" check. Run them in the pilot or a follow-up pack if v0 shows signal.
