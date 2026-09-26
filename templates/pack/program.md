# <pack> — instructions for the research agent
You make ONE change per call. The runner (not you) runs training/evaluation, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator or the data; do not try.
## Goal
<maximize|minimize> <metric>. Current noise: see sigma in history.md. A change is kept only if it beats the
incumbent on one seed and then clearly (by more than 2·SE) on fresh seeds.
## What you may change
<surface files>. Nothing else has any effect. Data and model calls come only through the frozen
harness/client; the surface must not read files, the environment or the network.
## What the code does (2–5 lines)
## Ideas worth trying (from IDEA.md)
## Rules
- One change, one hypothesis_tag (reuse an existing tag for the same idea).
- Read history.md: don't repeat a failed idea unless you change it materially. After 3 discards in a row, try something structurally different.
- Respect constraints.md (facts about this machine).
- Prefer simple changes; equal results with less code are better.
- Keep notes.md short and useful to your future self.
- Finish with the JSON object required by the output schema and nothing after it.
