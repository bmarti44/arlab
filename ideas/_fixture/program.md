# _fixture — instructions for the research agent
You make ONE change per call. The runner (not you) runs training/evaluation, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator or the data; do not try.
## Goal
maximize accuracy on a synthetic 10-class problem. A change is kept only if it beats the incumbent on one
seed and then clearly (by more than 2·SE) on fresh seeds.
## What you may change
model.py. Nothing else has any effect. The surface must not read files, the environment or the network.
## What the code does
The frozen harness calls build(), then train_step() for exactly 1500 steps of batch 1024, then predict().
## Rules
- One change, one hypothesis_tag. Read history.md; respect constraints.md. Prefer simple changes.
- Finish with the JSON object required by the output schema and nothing after it.
