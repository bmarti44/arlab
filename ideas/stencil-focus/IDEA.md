# stencil-focus — does the *selection* of re-presented instructions matter for a frozen model?

**Where the idea comes from.** Earl Miller's "mobile stencil" account of cortical waves (Picower article; Neuron
2023): stored patterns are one thing, and a moving stencil that picks which of them are read out at this moment is
another. `~/stencil-llm` (the owner's repo, `README.md`) took this as a split between stored knowledge and current
focus for a frozen chat model. Its results settle a lot: weight-side/attention-side steering did nothing or hurt
(static bias n=196 −4.6; router bias cut competence 16/32→7/32); a learned selector did not beat a role/recency rule
(C2, −3.5); and restating the correct current rules in prose at request time was the strongest mechanism in every
harness. Its final head-to-head (Exp 4C, `results/memorycode-long/RESULTS-4C.md`, Qwen3-4B, N=139) restated the
*evicted* mentor sentences newest-first in a 256-token reminder under a 3,584-token window: compliance 0.084→0.105,
+2.05 points [−0.07, +4.16], p=.058, NOT PROVEN. The closest prior is Exp 3B (`results/memorycode-derived/
setup-4b/summary.json`, 4B, 16 short items, full history dropped): restate_all .099, the FOCUS-3 `auto` selector
.203, oracle .255 — auto beat restate_all by +10.4 [−7.3, +31.3], then the experiment was declared INELIGIBLE on its
strict gate. The descriptive ceiling in the long regime (`RESULTS-4B.md`, n=16, research runtime, oracle clipped to
256 tokens on 8 items): oracle .263 vs .109 shipping policy vs .066 no reminder, oracle − focus +15.4 [+4.0, +27.3].
So stencil-llm has one label-free selector with a hint of an effect and a wide, small-n oracle gap; what it never
did is *search* over reminder policies (which old sentences, how many window tokens, order, placement) in the
long/evicted regime, and that search is exactly what arlab is for. None of this tests biology.

**Question (sized for arlab).** With the model, the prompt window and the evaluation frozen, can a *label-free*
selection/rendering policy over a long conversation's earlier sentences raise compliance with the coding
conventions in force by ≥ 5 points over verbatim newest-first restatement of the evicted region? Direction:
maximize. Hypothesis: yes — separating conventions from coding chatter plus supersession handling (newest statement
per convention wins) recovers a meaningful share of the oracle gap. A null ("no policy ≥ 5 points found in N
experiments", on a bank whose pilot showed ≥ 10 points of oracle headroom) would mean that on this workload the
selection layer is not where a frozen 4B model loses; the loss is downstream (using a correct reminder, output
failures), and the next lever is model- or format-side, not selection. The `evicted_1024` reference (below)
separates "select better" from "restate more".

**Metric.** `fraction_required` (stencil's Exp 4B/4C primary, `src/stencil/memorycode.py:fraction_required`): per
item, the equal-weight mean over the MemoryCode regex checks whose family the query requires, denominator frozen
before generation, 0 when the required class/function is missing; the vendored official checker
(`vendor/memorycode/code/evaluate_model_output.py`, unmodified) does the matching. Item pack, one item per
dialogue, `items` filled, `verdict: {compare_to: baseline}`. **Baseline** = the 4C shipping policy: every mentor
sentence lying before the *base* window's cut (the cut with no reminder), packed newest-first into 256 tokens under
"Earlier instructions still in force:", rendered chronologically after the truncated thread and before the request
(`evicted_mentor_sentences`, `pack_long`, `render_long_reminder`, `build_long_prompt`). **References** (frozen
surfaces under `frozen/run/`): `off` (no reminder) and `evicted_1024` (the baseline policy with a 1,024-token
budget). CALIBRATE then measures the per-item sd from the references instead of assuming √0.2, and the report can
attribute a win to selection rather than size. The baseline will not reproduce 4C's numbers (vLLM instead of the
research runtime, a different generation cap, a synthetic bank); that is expected and not a check.

**MES (fixed before calibration): 0.05.** Five points is 2.5× the +2 the 4C sample could not separate from zero and
a third of the (small-n) oracle gap; with the 4C paired sd (0.126) and 300 holdout items 2·SE ≈ 0.015, and with the
sd of oracle − focus on the 4B SETUP items (0.244) still ≈ 0.028 (≈ 0.031 at 250 items). Under the runner's assumed
sd 300 items give 2·SE = 0.052, so the references are not optional. Seeds: vLLM temperature 0 is not bit-repeatable,
so a seed pack `{calibration: [1, 2, 3], screen: 1, confirm: [2], holdout: [101, 102]}` (the harness ignores the
seed value; sigma is measured either way).

**Surface (the only editable file), `surface/stencil.py`.** `plan(sentences, current, query, window)` receives
the earlier sessions as records `{session, speaker, text, tokens, evicted_at_base_window}` (whole sentences from the
frozen splitter `stencil.focus3.sentences`, mentor/mentee names given), the current session's text, the request and
W. A frozen helper `frozen/run/window.py` (agent-visible) answers "which sentences are displaced at reminder budget
B", so the surface never re-implements the tokenize-and-retrim loop. It returns a **Plan**: an ordered list of
sentence ids, a header id from a frozen set (incl. none), placement (`before_thread` | `after_thread`) and a
reminder budget ≤ 1,024 tokens. The frozen harness builds the prompt exactly as `build_long_prompt` does: newest
thread tokens fill whatever the reminder leaves of W = 3,584; native single-user-message format with the empty
`<think>` block; the generation cap (512 / 768 / 1,024) is fixed from the pilot before sealing. Every reminder line
must be a whole verbatim sentence of sessions 0..s−1, or the run is `invalid` — the surface selects, orders, sizes
and places; it never writes. Query-conditioned selection (class-side vs function-side conventions) is legitimate
and program.md says so. Pure Python, stdlib plus the frozen helpers: no files, argv, env, network, no model or
embedder loads from `/hf`, no auxiliary LLM calls (a later tag may add them). A pack test checks that the baseline
surface reproduces `evicted_mentor_sentences` + `pack_long` byte-for-byte on sample bank items.

**Data — never stencil-llm's pools.** MemoryCode is exhausted for our purpose: of its 360 dialogues, 292 are in
stencil-llm's registered pools (`results/memorycode-derived/items.json`: 16 SETUP + 64 SCREEN;
`results/memorycode-long/items.json`: 16 SETUP-LONG + 128 SCREEN-LONG; `items-4c-candidates.json`: the 196 4C
candidates incl. the 68 reserve), 40 have no usable history item, 28 remain. Nothing under `data/bench/`
(Multi-IF, IFEval, IFBench, GSM8K, MMLU-Redux, BFCL), nothing from the candidate-A SCREEN/TRAIN pools
(`results/a-screen/*.json`, `src/stencil/a_screen_pool/`, `a_train_pool.py`), and none of the 29 oasst2 trees
stencil fit on (`data/g0/chat.jsonl`, `LEDGER-PLAN.md:442`; PREPARE reads it from the snapshot, drops those trees
by root id, a test asserts none remain). Instead PREPARE builds a **fresh MemoryCode-style bank**, deterministic
and LLM-free at PREPARE time, from `vendor/memorycode/topics.json` (51 pivots with update variants, regexes and
eval queries; 30 instruction-like non-coding "filler instructions"; 19 contexts; Apache-2.0), following
`code/generate_template.py`'s template logic: pivots introduced over time, updates only on the affix pivots 6–15 as
upstream, `history_regex` = latest regex per live pivot, query = a live pivot's `eval_query`, one item per dialogue
(the final session). Two design rules make selection non-trivial, as in real MemoryCode (`dialogue_186` combines
two conventions in one sentence; CONTRACT.md amendment 1 records that topic-string containment fails):
conventions are rendered through a **frozen paraphrase set** authored once by Codex in build tooling (§6.3
style) — several templates per family, compound sentences carrying two conventions, conventions embedded
mid-sentence, no marker frames — and the chatter keeps its programming vocabulary and gains **code-adjacent
distractors**: the 30 filler instructions with their updates, and Codex-authored mentor talk about code that is
not a convention. Chatter comes from oasst2 (minus the g0 trees) and `HuggingFaceH4/ultrachat_200k` `test_sft`
(MIT, revision `8049631c`, downloaded in PREPARE; assistant → mentor, user → mentee), enough for ~9–12 M tokens;
dialogues have 12–30 sessions and 8–24 k-token histories (real long items: median 15.6 k), so the evicted region is
most of the history, and the current session always fits. Threads are never reused; chatter is never scored.
**Split:** 300 validation / 300 holdout dialogues (250/250 if the pilot says so), seeded; pivots split disjointly
and stratified (affix pairs across sides 6|11, 7|12, …; multi-pivot families spread; singleton families assigned
and recorded), paraphrase templates and chatter threads disjoint too; a test checks similar update counts and
family mixes. Public = session texts and the request; private = live sets, `history_regex`, `required`,
`structure`, `topics.json`. **Ceiling test** in `tests/`: a frozen trivial keyword/frame selector scored against
the live set must reach F1 < 0.8 on both splits, else the bank is regenerated. **Construction details** (templates,
frames, distractor recipes, exact sampling) live in `frozen/prepare/BANK.md`, not here, and `agent.visible` is
`[program.md, frozen/run/window.py]` — the agent does not see this file. Descriptive only, in the holdout: the 28
free real MemoryCode dialogues re-scored at W = 1,536 (20 of them have evicted history there), never the primary or
a guard. `splits.json` written by PREPARE.

**Pre-seal pilot (go/no-go).** `build/pilot.sh` on a separate 60-dialogue pilot slice (in neither split, chatter
and templates disjoint, both pivot sides): arms `off`, baseline, `evicted_1024`, and a label-derived oracle
(≤ 1,024 tokens). Measures oracle − baseline, paired sd, output-failure and cap-hit rates, aggregate decode tok/s and
minutes per 100 items. **Build only if oracle − baseline ≥ 0.10 (2×MES) and a 300-item run projects to ≤ 20 min**
(else 250/250, then reconsider). The pilot fixes the generation cap and `run.timeout_s` (2× the projection).
Result recorded in DECISIONS.md.

**Use of stencil-llm code.** Path-limited pinned snapshot in PREPARE, SHA pinned in `prepare.py`:
`git -C ~/stencil-llm archive 87b72a88cd56712a865359afd794a34b66f3d05d src/stencil/{__init__,memorycode,focus3,
focus2,stats}.py vendor/memorycode data/g0/chat.jsonl | tar -x` (HEAD 2026-09-13; `focus3` imports `focus2` which
imports `stats`). The five modules go to the RUN-visible data (they need no `vendor/`); `vendor/memorycode` goes to
`private/` only (EVALUATE calls `vendored_checker`); `data/g0/chat.jsonl` is used for the exclusion and not kept.
Nothing else is extracted, nothing is edited; stencil's FROZEN files are not touched. GPL-2.0, local use only (the
pack folder says so). Model weights are not in the archive.

**Model and resources.** Qwen3-4B, the 4C trunk: `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`
(`deploy/stencil_focus/MODEL_CARD-4b.md`; local copy `~/stencil-llm/models/qwen3-4b-hf`). The HF cache holds only a
stub, so PREPARE downloads it (~8 GB); vLLM 26.04 serves it as in `memory-longmemeval` (raw-prompt completions,
temperature 0, `max_model_len` 8192, `--max-num-seqs 64`, prefix caching, `mem_gb` ≈ 30). Timing, to be measured
in the pilot: this machine's aggregate decode was 160–350 tok/s at 16–64 streams (DECISIONS.md), so 300 items ×
~600 decode tokens plus ~1.1 M prefill tokens is 12–20 min per run, not minutes. **Budget:** `service_tokens ≤ 300 ×
(3,584 + cap) ≤ 1,382,400` per split. Campaign: `max_hours` 12, `max_experiments` 40, `stop_after_no_keep` 15;
expect 20–30 experiments (calibration 3 + 2 reference runs, a confirm run per screen pass, 4 holdout runs).

**Guards and cheating risks.** Guards: `output_failure_rate` (union of no-parsable-code / hit-cap / 4-gram
repetition > 0.5 / timeout per item, as in `output_failures`) `max_ratio_vs_baseline: 1.15` — 4C saw 0.43 in both
arms under its own setup, and a policy that trades compliance for broken code must not win — and `train_s`
`max_ratio_vs_baseline: 1.3`. The model is frozen, so general capability cannot regress; no MMLU/GSM8K guard.
Risks defended: (1) label leakage — RUN sees only text, the request and the eviction flags; live sets, regexes and
`topics.json` are private; (2) hard-coded conventions — no free text anywhere in the reminder (verbatim whole
sentences + frozen headers, checked by the harness and re-checked by the evaluator from the recorded prompt), and
EVALUATE marks the run `invalid` if the surface source contains any ≥ 5-word n-gram from `topics.json` instruction
texts or eval queries (program.md states the rule; the paraphrased bank makes such lists useless anyway);
(3) window cheating — the evaluator re-tokenizes every recorded prompt; > W or > cap generated tokens → `invalid`;
(4) budget — all model calls go through the budgeted client, `budget.json` over the limit → `invalid`;
(5) construction artifacts — paraphrase templates, chatter and pivots disjoint across splits, construction details
hidden from the agent, ceiling test. Tests: the scorer on hand-written outputs (required parent missing → 0;
optional families ignored), the verbatim check on a paraphrased line, the window check on an over-long prompt,
every inserted convention being exactly one `focus3.sentences` span (the splitter breaks on abbreviations), the g0
exclusion, the split stratification, the ceiling selector, and one bank dialogue reproduced end to end (template →
text → live regexes equal the checker's expectations).
