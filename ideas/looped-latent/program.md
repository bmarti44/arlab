# looped-latent — instructions for the research agent
You make ONE change per call. The runner (not you) runs training/evaluation, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator or the data; do not try.
## Goal
maximize accuracy: exact-match of the greedy answer (≤ 6 tokens, no written reasoning) to "what does this short
Python program print?" (1–6 dependent statements: + and − with small constants, swaps, ifs; values mod 100).
Current noise: see sigma in history.md. A change is kept only if it beats the incumbent on one seed and then
clearly (by more than 2·SE) on two fresh seeds.
The question: does latent computation in the hidden state, i.e. rerunning a middle block of layers K times before
emitting a token, beat the same model fine-tuned without a loop under the same wall-clock budget? Everything except
the loop is frozen and identical for every arm, so any keep is a loop effect.
## What you may change
loop.py only, and only the loop machinery: the loop window (LOOP_START/LOOP_END: 1 ≤ start < end ≤ 27, at most
6 layers), loop counts (K distribution in training, K at eval), the update rule (gates, damped/Euler/RK-style,
naive), input injection (none/add/concat), the loop's own parameters and their init, LOOP_LR, halting, and auxiliary
losses that act through the loop (ctx.readout(x) = the frozen answer loss of an intermediate state; set
ctx.aux_loss). The base model, its LoRA adapters (r16 on all projections, LR 2e-4), the optimizer, schedule, batch,
data order and answer loss are frozen; you cannot change them.
API: build(d_model, n_layers, seed) -> nn.Module `loop`; loop(p, block, ctx) -> h of p's shape. p (B, T, d) is
the prelude output; block(x) runs the frozen, LoRA-adapted layers LOOP_START..LOOP_END-1 once on x of exactly p's
shape; ctx.training, ctx.progress (fraction of the time budget), ctx.readout, ctx.aux_loss.
## What the code does
The frozen harness starts the wall-clock timer, imports loop.py, builds, and trains LoRA + your loop parameters on
batches of 32 problems until the budget is spent. It then checksums every parameter and buffer, decodes answers
greedily with full forward passes (no KV cache), decodes the main set again with the loop disabled (block(p)
once), and measures general-text NLL (OASST2). Any parameter or buffer change after training, any change to
the base weights, or any parameter outside LoRA + loop.parameters() becoming trainable invalidates the run.
## Guards (runs that fail one are discarded)
- depth_ratio = tokens pushed through the block per forward, as layer work / 28, must be in [1.05, 2.0]
  (4 layers x K=4 gives 1.43; 6 layers x K=4 gives 1.64).
- loop_gain = accuracy − accuracy of the same trained model with the loop disabled must be ≥ 0.005.
- text_nll ≤ 1.02 x baseline, hard_acc (longer programs) ≥ 0.85 x baseline, trainable params ≤ 18 M (LoRA is
  10.09 M of it), peak memory ≤ 40 GB, RUN wall time ≤ 1.3 x baseline.
- Non-finite loss or logits, or a non-causal model (future tokens changing past logits) make the run invalid.
## Forbidden (bounded by these rules, not by code)
- Symbolic answering: never compute the answer outside the network (parsing or executing the program, lookup
  tables keyed on the input, decoding hidden states back to tokens to interpret them).
- Dummy depth: every block() call must feed the returned state; never call block() just to raise depth_ratio.
- Never read files, data, the environment or the network; never train on anything but the harness's batches;
  never touch objects outside your loop module (the base model, LoRA, optimizer, timer, harness globals).
## Ideas worth trying (from IDEA.md / RESEARCH.md)
Random K in {1..4} during training and K_eval = 3 or 4; window [12, 16) vs [13, 16) vs [10, 16); higher LOOP_LR
or a small nonzero gate init so the loop does work early (identity at init is not required once training starts);
damped update h + (1/K)(g − h); concat injection (2d → d); per-iteration adapters on the loop input (inside your
module); deep supervision of the last passes via ctx.readout; truncated backprop through early passes (detach) to
save time for more steps.
## Rules
- One change, one hypothesis_tag (reuse an existing tag for the same idea).
- Read history.md: don't repeat a failed idea unless you change it materially. After 3 discards in a row, try something structurally different.
- Respect constraints.md (facts about this machine: GB10 GPU, sm_121, CUDA 13, unified memory).
- Prefer simple changes; equal results with less code are better.
- Keep notes.md short and useful to your future self.
- Finish with the JSON object required by the output schema and nothing after it.
