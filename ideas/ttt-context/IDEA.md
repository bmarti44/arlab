# ttt-context — learning a long context into the weights at test time (TTT track, stage 0)

**Source.** Stage 0 of `ideas/plastic-agent/TTT-DEEP-DIVE.md` §4 ("keep the memory brief's E1, with changes") and
`ideas/memory-architecture/RESEARCH.md` §5 E1 (qTTT on Qwen3-1.7B, query projections only, generated ~16K-token
retrieval set, 400 validation / 400 holdout). Adjustments taken from the deep dive: the LongMemEval-16K secondary is
dropped (its expected 0–4 points is below any MES); time is calibrated first and the context is cut to 8K or the steps
to 16 if a run is too slow; span selection is the main lever left to the agent.

**Question at this scale.** With a frozen per-item wall-clock budget of a few seconds, does test-time training of
Qwen3-1.7B on a long document (weights reset for every item) answer questions about that document better than plain
long-context answering with the same document in context (the no_ttt arm)? The published anchor is qTTT (arXiv
2512.13898): W_Q-only updates with the KV cache frozen after one prefill, +12.6 / +14.1 points at 4B, also tested at
1.7B. S-TTT (2607.09415): random spans can *hurt*, oracle spans help. Prior (RESEARCH.md): ~40 % that a candidate
clears the MES on a synthetic set. This stage also de-risks the TTT loop, the per-item reset and the timing for
stage 1 (`plastic-agent`); its effect size does not gate stage 1.

**Model.** Qwen3-1.7B (cached snapshot `70d244cc`, bf16, 28 layers, d = 2048, 16 query / 8 KV heads), thinking
disabled (empty `<think></think>`), the assistant turn prefilled with `Answer:` (a CPU check on 12 items: without it
the base model often continues the log or starts explaining instead of answering).

**Data (frozen generator in `frozen/prepare/`, mounted in PREPARE only; no LLM, no downloads).** Each item is one document and one question.
A document is the operations log of a fictional depot, ~1,100 one-line timestamped events over several days: crates
moved between docks, shipments approved/rejected/reviewed, people assigned to teams, teams moving rooms, locker codes
set, and routine filler. All names are pseudo-words drawn per document, so answers cannot come from world knowledge.
Four question kinds, balanced (kinds cycle state/hop2/count/kv):
- *state*: where a crate is at the end (last of 3–5 moves; near-duplicate crate codes with their own moves);
- *hop2*: the room of a person's team at the end (person → last team → that team's last room; a namesake is a decoy);
- *count*: how many shipments a person approved (1–6, among rejections/reviews by the same person and approvals by namesakes);
- *kv*: the latest code of a locker (last of 1–3 settings; near-duplicate locker names with their own codes).
Answers are 1–6 tokens (a dock name, a room number, a count, a code). Each document is fitted to at most
`CONTEXT_TOKENS` = 8,192 tokens (chat head + log; within 128 tokens of it; pilot: 16K made no_ttt 0.175 and runs
~24 min, so 8K was chosen) by removing random unprotected distractor events; the answer is recomputed from the final
event list. Validation and holdout come from disjoint generator seeds, 400 items each (`N_ITEMS`). The generator, its
seeds, the sizes and the difficulty knobs (`DIFFICULTY`) live in `frozen/prepare/`, which RUN never mounts, so no RUN
code can regenerate the documents or their answers (astra review, round 1).

**Harness (frozen, `frozen/run/tttlib.py`).** Two copies of the model: ANSWER (never handed to surface code) and
WORK (the surface's working copy). Per item: reseed every RNG from the item seed; prefill the document on ANSWER with
the pristine weights (not counted as TTT, identical in every arm) and give the surface its own clone of that cache;
then, timed between device syncs, import `ttt.py` afresh, call `adapt(WORK, ctx)`, and validate and clone its return
value `{"weights": {name: tensor}, "doc_in_context": bool}` (literal bool; known names, exact shapes/dtypes, finite).
Before answering, the harness restores the process state surface code could have touched (global and per-module
hooks, the class dictionaries of the model's module classes, the dictionaries of the modules the answer path uses,
the attention/mask registries, sys.modules entries the surface added; a thread left running is refused), checks that
ANSWER is still pristine, copies in only the returned weights and answers greedily (≤ 12 tokens) from the frozen
prompt: the stored question tokens after the harness's own document cache, or after the chat head alone. No surface
code runs while answering, and a surface cannot supply caches or prompts. Afterwards ANSWER and WORK are restored
(weights, buffers, hooks, instance attributes, config; structural changes or a monkeypatched `forward` crash the run)
and verified equal to the pristine weights, plus a probe-logit check (within 0.25, bf16 noise). The SHA-256 of both
models at the end must equal the pristine weights' at load.

**Arms.**
- *no_ttt* (frozen reference `frozen/run/ref_no_ttt`, the verdict's comparison): document in context, no updates.
- *no_doc* (frozen `frozen/run/ref_no_doc`, pilot only): the question alone — a floor (expected ≈ 0 except counts,
  ~1/6 on a quarter of items, so ≤ ~5 %). Not a `pack.yaml` reference: the runner's power check takes the largest
  paired item sd over references, and no_doc vs any arm that reads the document has sd ≈ 0.5, which would mark the
  pack underpowered by construction.
- *baseline surface* (`surface/ttt.py`): the simplest TTT — next-token loss on random 128-token document spans,
  updating only `q_proj` of all 28 layers, 8 Adam steps (fp32 master copies, LR 1e-4), keys/values of the document
  before each span frozen from the harness's prefill (qTTT's mechanics), document kept in context. **Choice:** qTTT's
  mechanics are in the baseline, its larger budget (32 steps) and any span selection are left to the agent, so the
  baseline is "naive TTT" (S-TTT predicts ≈ no_ttt or worse) and the agent's job is the part the literature says
  matters. Because the verdict compares against no_ttt, the baseline's quality does not bias the verdict.

**Metric.** Exact-match accuracy over the 400 items (item pack, one 0/1 per question): the evaluator decodes the
generated token ids itself, takes the first line up to the stop token, normalizes (lowercase, punctuation → space,
articles dropped, number words → digits) and compares with the gold aliases (`Velm` / `dock Velm`, `214` / `room 214`,
`3`, `48213`). Hedges, sentences and lists score 0. An item whose TTT time exceeded the budget scores 0.

**MES (fixed before calibration): 0.05.** Holdout: 400 items × 2 seeds, compared with no_ttt. The owner's planning
formula SE = √(p(1−p)/n + 2σ²/2) gives SE ≈ 0.025 at p = 0.3, σ = 0.01 (2·SE = 0.050) and 0.027 at p = 0.5
(2·SE = 0.054): 5 points is the smallest effect this set resolves. The runner's power check uses the paired item sd
measured between no_ttt and the baseline (sd 0.3 → SE 0.018; sd 0.4 → 0.022; with σ = 0.01) and is expected to pass.
If it does not, raise `N_ITEMS` under a new tag, never the MES. Seeds: calibration [1, 2, 3], screen 1, confirm [2],
holdout [101, 102] (TTT is stochastic through the span choice, so σ > 0).

**Budget (calibrate in the GPU pilot, then fix).** `--ttt-seconds 4` per item (qTTT-like 32 steps × 128 tokens with a
16K prefix is estimated at 3–6 s on the GB10), `budget: {unit: ttt_seconds, limit: 1600}` = 400 × 4. A run is
≈ 60 s + 400 × (prefill ≈ 2 s + TTT ≤ 4 s + answer ≈ 0.3 s) ≈ 45 min. If that is too slow, cut `CONTEXT_TOKENS` to
8192 (prefill ≈ 0.8 s) and/or `--ttt-seconds` to 2–3; keep `N_ITEMS` = 400 (power). Guard: `peak_mem_gb ≤ 40`.
No `train_s` ratio guard: the per-item time budget is the compute budget.

**Pilot gate (before the full `arlab check`).** no_ttt at 30–70 % on validation (else tune `DIFFICULTY` or the
context length), no_doc ≤ 8 %, a timing table (prefill_s, ttt_s per step, answer_s) at 16K and 8K, σ from baseline
seeds 1–2, and one qTTT-like arm (32 steps, LR 1e-3) to see whether any headroom exists at 1.7B.

## Open risks (for REVIEW.md)
1. **Program-aided shortcuts.** The surface sees the document tokens and the question. It could parse the synthetic
   log with templates and train the model to emit the parsed answer, which would raise the score without the intended
   mechanism. `program.md` forbids hand-written parsing and code-built targets (targets must be document text or the
   model's own generations); the agent does not see the generator or the data; diffs are in the report. Generic
   question–span token overlap for span selection is allowed on purpose (it is the S-TTT lever).
2. **In-process surface.** As in every arlab training pack, `adapt()` runs inside the harness process. The answer
   path is protected by a separate ANSWER model, a weights-only return and the state restore above, but deliberate
   patching of objects the harness does not snapshot (e.g. `torch` itself or the timer) or reading `/data/public`
   (other items' documents; never answers) remains possible. Forbidden by `program.md`, visible in every kept diff;
   the proposer is good-faith (PLAN §3.7). A subprocess per item was rejected: it reloads 1.7B weights per item.
3. **Wall-clock budget on a shared GPU.** Throughput changes shift how many steps fit; σ from 3 seeds absorbs part of
   it and contended runs are re-run. The surface should leave a margin (the baseline stops 0.3 s early).
4. **Power check optimism.** If the naive baseline barely changes answers, the measured no_ttt-vs-baseline item sd
   is small and the planned SE optimistic; the verdict still uses the observed holdout SE.
5. **Format vs knowledge.** Next-token TTT on the log can teach the model to continue log lines instead of answering
   (seen on CPU before the `Answer:` prefix). The first-line scoring and the prefix limit this; a surface can also
   learn format only. Stage 1's placebo arm addresses this for agents; here the no_ttt comparison is per item.
6. **Slow experiments.** ~45 min per run at 16K and 400 items is 4× arlab's 10-min target: ~15 experiments per
   24 h. The pilot decides between 16K and 8K.
