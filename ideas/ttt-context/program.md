# ttt-context — instructions for the research agent
You make ONE change per call. The runner (not you) runs the experiment, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator or the data; do not try.
## Goal
maximize accuracy: exact match of Qwen3-1.7B's short greedy answer to one question about one long (~16K-token)
synthetic document, after test-time training (TTT) on that document within a fixed per-item wall-clock budget.
Documents are logs of fictional events; questions need state tracking (latest value after several updates),
two-hop lookups, counting, or picking the right entity among near-duplicates. Answers are 1-6 tokens.
The verdict compares the final surface with the frozen no_ttt arm (document in context, no updates) on a holdout.
A change is kept only if it beats the incumbent on one seed and then by more than 2·SE on a fresh seed.
## What you may change
ttt.py only: which parameters adapt (q_proj, k_proj/v_proj, o_proj, norms, MLPs, a low-rank delta merged into
weights, a subset of layers), the loss (next-token on spans, spans weighted by relevance, a query-conditioned
loss, the model's own generated question/answer pairs about the document, self-distillation), span selection
(random, overlap with the question's tokens, attention of the question to the document, highest-loss spans),
steps, span length, learning rate, optimizer, schedule, whether the document stays in context, re-encoding the
document with the adapted weights (return that cache), and early stopping against ctx.time_left().
Keep the API: adapt(model, ctx) -> None | Cache; constants DOC_IN_CONTEXT, PREFILL_DOC.
## What the code does
For every item the frozen harness imports ttt.py afresh, prefills the document with the base weights, then calls
adapt(model, ctx). ctx has doc_ids (1, L), question_ids, doc_len, device, generator (per-item seed), time_left()
and prefix_cache(n) (base-model keys/values of document positions < n; do not write into it in place).
The baseline updates q_proj of all 28 layers with 8 Adam steps (fp32 master copies, LR 1e-4) of next-token loss
on random 128-token spans, reading the frozen base keys/values before each span (qTTT mechanics), then answers
with the document in context. After adapt(), the harness answers greedily (12 tokens max, the assistant turn
starts with "Answer:"), then restores every weight, buffer and hook, and verifies the restore.
## Budget and guards
- Time from the import of ttt.py to the end of adapt() (device synced) must be <= the per-item budget
  (currently 4 s). An item over budget scores 0. Check ctx.time_left() before each step.
- Change parameter values only (in place). New or replaced modules/parameters, or a monkeypatched forward(),
  crash the run. Forward hooks are allowed; the harness removes them after each item.
- Peak memory <= 40 GB. Non-finite logits or a failed weight restore make the run invalid.
## Ideas worth trying (from IDEA.md)
Question-aware span selection (spans that share rare tokens with the question, or that the question's queries
attend to most) instead of random spans (random spans can hurt: S-TTT); more steps with shorter spans; a higher
LR for q_proj; adapting k_proj too and re-encoding the document; a query-conditioned loss (the question
followed by a candidate span); training on the model's own generated Q/A pairs about the document; fewer layers
(upper half) to save time for more steps; keeping answers terse (a loss that does not teach log continuation).
## Rules
- Every training target must be text from the document or text the model itself generated. Never write code
  that parses the document's contents (templates, regular expressions, entity or number extraction) or that
  constructs an answer: selection may use generic token statistics, losses and attention only.
- Never read files, data, the environment or the network from ttt.py; use only what ctx gives you.
- Nothing may carry over between items (the module is re-imported and the model is reset anyway).
- One change, one hypothesis_tag (reuse an existing tag for the same idea).
- Read history.md: don't repeat a failed idea unless you change it materially. After 3 discards in a row, try something structurally different.
- Respect constraints.md (facts about this machine: GB10 GPU, sm_121, CUDA 13, unified memory).
- Prefer simple changes; equal results with less code are better.
- Keep notes.md short and useful to your future self.
- Finish with the JSON object required by the output schema and nothing after it.
