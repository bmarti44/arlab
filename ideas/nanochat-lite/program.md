# nanochat-lite — instructions for the research agent
You make ONE change per call. The runner (not you) runs training/evaluation, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator or the data; do not try.
## Goal
minimize val_bpb (validation bits per byte). A change is kept only if it beats the incumbent on one seed
and then clearly (by more than 2·SE, see history.md for sigma) on two fresh seeds.
## What you may change
train.py only: the model, optimizer, schedules, hyperparameters, gradient accumulation. Nothing else has
any effect. Data comes only through the frozen harness; train.py must not read files, the environment or
the network, and must keep the API: build(config), train_step(state, batch, step, total_steps) -> loss,
save(state, path), load(path, device) -> model whose forward(idx) returns logits (B, T, V).
## What the code does
The harness calls train_step once per 64×1024-token batch until the fixed token budget is spent
(total_steps calls); train_step accumulates gradients and steps the optimizer every TOTAL_BATCH_SIZE tokens.
The evaluator loads the checkpoint with load(), feeds tokens and computes cross-entropy from the returned
logits itself; the model must be causal (it is checked). Runs that take >1.05× the baseline's wall time (compute-matched),
exceed 60 M params or 60 GB peak memory fail their guards.
## Ideas worth trying
Learning rates and schedule shape, warmup/warmdown, batch size per optimizer step, width/depth/aspect
ratio at similar compute, attention/MLP variants, normalization, initialization, value embeddings,
logit softcap, Muon settings, weight decay.
## Rules
- One change, one hypothesis_tag (reuse an existing tag for the same idea).
- Read history.md: don't repeat a failed idea unless you change it materially. After 3 discards in a row, try something structurally different.
- Respect constraints.md (facts about this machine: GB10 GPU, sm_121, CUDA 13, no FlashAttention-3).
- Prefer simple changes; equal results with less code are better.
- Keep notes.md short and useful to your future self.
- Finish with the JSON object required by the output schema and nothing after it.
