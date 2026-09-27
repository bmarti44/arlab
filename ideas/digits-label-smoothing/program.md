# digits-label-smoothing — instructions for the research agent
You make ONE change per call. The runner (not you) runs training/evaluation, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator or the data; do not try.
## Goal
maximize test accuracy of a small classifier on 8×8 sklearn digits (64 features in [0, 1], 10 classes).
A change is kept only if it beats the incumbent on one seed and then clearly (by more than 2·SE) on fresh seeds.
## What you may change
model.py only: build(config) -> state, train_step(state, x, y, step, total_steps), predict(state, x) -> logits.
Nothing else has any effect. Data comes only through the frozen harness; the surface must not read files,
the environment or the network.
## What the code does
The harness runs exactly 1,500 train_step calls with random batches of 64 (torch, CPU, 1 thread), then
predict() on the evaluation inputs. The baseline is a 64-64-10 ReLU MLP with Adam (lr 1e-3), plain cross-entropy.
Runs that take > 1.5× the baseline's wall time fail a guard.
## Ideas worth trying (from IDEA.md)
Label smoothing; hidden width/depth; dropout / weight decay; learning-rate schedule; input normalization.
## Rules
- Never load the digits dataset (or any data) yourself: sklearn.datasets is blocked, and looking labels up is cheating.
- One change, one hypothesis_tag (reuse an existing tag for the same idea).
- Read history.md: don't repeat a failed idea unless you change it materially. After 3 discards in a row, try something structurally different.
- Respect constraints.md. Prefer simple changes; equal results with less code are better.
- Keep notes.md short and useful to your future self.
- Finish with the JSON object required by the output schema and nothing after it.
