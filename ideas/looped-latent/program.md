# looped-latent — instructions for the research agent
You make ONE change per call. The runner (not you) runs training/evaluation, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator or the data; do not try.
## Goal
maximize accuracy: exact-match of the greedy answer (≤ 6 tokens, no written reasoning) to "what does this short
Python program print?" (2–8 dependent statements, values mod 100). Current noise: see sigma in history.md. A change is
kept only if it beats the incumbent on one seed and then clearly (by more than 2·SE) on two fresh seeds.
The question: does latent computation in the hidden state, i.e. rerunning a middle block of layers K times before
emitting a token, beat the same model fine-tuned without a loop under the same wall-clock budget?
The baseline is that no-loop arm (LOOPS_TRAIN = (1,), LOOPS_EVAL = 1); a keep must loop (see guards).
## What you may change
loop.py only: the loop window (LOOP_START/LOOP_END, at most 6 layers), loop counts (LOOPS_TRAIN distribution,
LOOPS_EVAL), the update rule (gate, damped/Euler/RK-style, naive), input injection (none/add/concat), per-iteration
adapters on the loop block, LoRA rank/targets, extra supervision (e.g. deep supervision of intermediate passes
through the coda), learning rates and schedule, truncated backprop through passes, a halting rule.
Keep the API: build(base, config) -> state with state["model"] the nn.Module holding every parameter;
train_step(state, ids, targets, progress) -> float loss; logits(state, ids) -> (B, T, V) logits.
Call decoder layers as modules (layer(h, ...)); do not copy or re-implement them.
## What the code does
The frozen harness loads Qwen3-0.6B (bf16, 28 layers), starts a wall-clock timer, calls build(), then calls
train_step on right-padded batches of 32 training problems (targets = answer tokens, -100 elsewhere;
progress = elapsed / budget) until the time budget is spent. It then decodes answers greedily with full forward
passes through logits() (no KV cache), and measures the per-token NLL of general text (OASST2) with logits().
## Guards (runs that fail one are discarded)
- depth_ratio = decoder-layer calls per forward / 28 must be in [1.05, 2.0]: the eval forward must really loop,
  and inference may cost at most 2x the base (4 layers x K=4 gives 1.43; 6 layers x K=4 gives 1.64).
- text_nll ≤ 1.02 x baseline (do not damage general language modelling), hard_acc (9–12-step problems) ≥ 0.88 x
  baseline, trainable params ≤ 18 M, peak memory ≤ 40 GB, RUN wall time ≤ 1.3 x baseline.
- Non-finite loss or logits, or a non-causal model (future tokens changing past logits) make the run invalid.
## Ideas worth trying (from IDEA.md / RESEARCH.md)
Random K in {1..4} during training and K_eval = 3 or 4; window [12, 16) vs [13, 16) vs [10, 16); higher LOOP_LR
or a small nonzero gate init so the loop does work early (check that gates actually open); damped update
h + (1/K)(g − h); concat injection (2d → d); per-iteration LoRA on the loop block; deep supervision of the
last two passes; truncated backprop through the first passes to save time for more steps.
## Rules
- Never read files, data, the environment or the network from loop.py; never train on anything but the batches
  the harness hands you; never look for answers. Data and evaluation are frozen and hidden.
- One change, one hypothesis_tag (reuse an existing tag for the same idea).
- Read history.md: don't repeat a failed idea unless you change it materially. After 3 discards in a row, try something structurally different.
- Respect constraints.md (facts about this machine: GB10 GPU, sm_121, CUDA 13, unified memory).
- Prefer simple changes; equal results with less code are better.
- Keep notes.md short and useful to your future self.
- Finish with the JSON object required by the output schema and nothing after it.
