# IDEA-REVIEW — stencil-focus (review of IDEA.md as of 2026-09-27)

Scope: faithfulness to the stencil article and stencil-llm results, PLAN §6.4 hard constraints, statistics,
feasibility on the GB10, cheating risks, and build blockers. Everything was checked read-only against
`~/stencil-llm` at HEAD `87b72a88` (the SHA IDEA.md pins). Small CPU counts were run from /tmp.

## What checks out

- **4C numbers** (`results/memorycode-long/RESULTS-4C.md`): off 0.0845, on 0.1049, +2.05 points
  [−0.07, +4.16], p = .058, N = 139, SD of D = 12.59 points (so "paired item sd 0.126" is right), failure
  rate 0.432 / 0.439. All correct.
- **4B numbers** (`RESULTS-4B.md`): oracle .263, focus .109, base .066, n = 16. Correct (see M7 for caveats).
- **Negatives**: static bias −4.6 (n = 196), router bias 16/32→7/32, C2 −3.5. These match `README.md`.
- **Pool accounting**: derived `items.json` covers 80 dialogues. The long pools (`items.json` 144 plus
  `items-4c-candidates.json` 196) cover 212. The two sets are disjoint, so 292 of 360 are used. That leaves 68 free:
  28 are short-eligible leftovers (108 − 80), and 40 have no usable item (30 single-session, 10 with no earlier
  instruction). This matches IDEA.md.
- **Code references exist**: `fraction_required`, `evicted_mentor_sentences`, `pack_long`, `render_long_reminder`,
  `build_long_prompt`, `output_failures`, `focus3.sentences`, and the vendored checker.
- **Cache and snapshot**: the HF cache holds only a 12 KB stub for Qwen3-4B. The oasst2 file is cached.
  HEAD = 87b72a88, dated 2026-09-13.
- **Power arithmetic**: 2.5 · 0.126/√300 = 0.018, and 2 · √(0.2/300) = 0.052. Both are right. The runner takes the
  measured sd from the references (`arlab/campaign.py:409-414`, `stats.expected_holdout_se`), so the `off`
  reference does make the pack pass the power check.
- **Faithfulness in broad terms**: stencil-llm's stated objective is "automatic maintenance of correct reminders"
  (`README.md`, `results/CURRENT-GOAL.md`). A label-free selection policy under a fixed window is the direct,
  arlab-sized form of that objective. The Miller "stencil = read-out selection" framing is used as an analogy only,
  and IDEA.md says so.

## BLOCKING

**B1. As specified, the synthetic bank makes the question trivial, and IDEA.md tells the agent how the bank is built.**
Three things combine:
- Instructions are verbatim `topics.json` strings placed in "a few fixed frames".
- Chatter has all programming vocabulary removed.
- Update variants differ only in the quoted affix. Pivots 6–15, for example, are "always start function names with
  'a_'" and its 7 siblings.

Together these mean a keyword or frame matcher reproduces the label-derived live set, and supersession becomes a
string diff. Real MemoryCode is not like this:
- Mentor lines paraphrase conventions and combine them. `dialogue_186` session 2 has "start method names with 'x_'
  and function names with 'fn_'". `results/memorycode-derived/CONTRACT.md` amendment 1 records that 6-word topic
  containment fails because of this paraphrasing.
- Chatter is coding-adjacent ("clean and consistent code", "never use a linter"). This is where stencil's selectors
  failed: FOCUS-3 v8 made 25 false admissions in 21 episodes (`README.md`).

A positive verdict on this bank would therefore be built in by the construction. That is the same criticism
stencil's README makes of its own GPT-2 result ("the gap is guaranteed by the construction"). A null result would say
nothing about real data either.

Separately, `agent.visible` includes IDEA.md by default. Its Data section gives the agent the exact generating
process, so policies can target construction artifacts that are shared by both splits. A pivot-disjoint holdout
cannot catch that.

**Fix:**
- (a) Render each convention through a frozen paraphrase set, several templates per family. Include compound
  sentences (two conventions in one) and conventions embedded mid-sentence, with no frame tokens that mark
  instructions. Author the set once with Codex, as §6.3 allows for task generation, and split it disjointly like the
  pivots.
- (b) Do not scrub programming vocabulary from the chatter. Keep MemoryCode's 30 `fillers_instruction` items and
  their updates as distractors, and add mentor talk about code that is not a convention.
- (c) Add a CPU test in `tests/` that reports a frozen trivial keyword selector's precision and recall against the
  live set. Fix a ceiling in advance (for example F1 < 0.8); if the selector beats it, the bank is regenerated.
- (d) Set `agent.visible: [program.md]`, or give the agent a copy of IDEA.md without the bank-construction details.

**B2. The oasst2 chatter cannot fill 600 disjoint dialogues.**
I counted the code-free English trees (the "strict" filter uses fences plus a programming word list; "mild" uses
fences plus python/javascript/sql/…):

| filter | trees | one branch per tree | every message, siblings included |
|---|---|---|---|
| strict | 2,731 | ≈1.2 M tokens | ≈4.0 M tokens |
| mild | 3,980 | ≈1.9 M tokens | ≈6.6 M tokens |

"Every message" repeats sibling answers to the same prompt, so it overstates the usable supply. The requirement:
- The ≥ 2×W floor needs 600 × ≥ 7,168 ≈ 4.3 M tokens.
- Matching real long items (median history 15.6 k, mean 19.9 k tokens, sessions 4–99, per
  `items-4c-candidates.json`) would need about 9–12 M tokens.

IDEA.md's "10–14 sessions" is also much shallower than the real items, so the evicted region would be smaller and
selection would matter less.

**Fix:** pick one of the following, and restate the power numbers:
- allow chatter threads to be reused across dialogues within a split, while keeping splits disjoint (the chatter is
  never scored);
- add a second chat corpus downloaded in PREPARE;
- shrink to 200/200 dialogues.

Aim for a length distribution closer to the real long cohort.

**B3. Hard constraint (§6.4): oasst2 is a stencil fit/select-on (TRAIN) pool.**
- `data/g0/chat.jsonl` is git-tracked, so it is inside the pinned archive. It holds 30 oasst2 branches from 29
  English trees.
- `LEDGER-PLAN.md:442` states "fit/select-on = OASST2 … under data/g0/", and `scripts/g0_oracle.py:44-46,196-202`
  builds that pool.
- IDEA.md does not exclude these trees, and a random draw can include them.

**Fix:** PREPARE reads `data/g0/chat.jsonl` from the snapshot and drops those 29 trees (by root `message_id`), with
a test that asserts none are present.

**B4. Nothing shows the synthetic bank has headroom, so a null would not support the reading IDEA.md gives it.**
- The 15-point oracle gap comes from 16 real SETUP-LONG items, with interval [+4.0, +27.3]. It was measured on the
  research runtime, with the oracle clipped to 256 tokens on 8 of the 16.
- On the new bank, neither the baseline level nor the oracle headroom is known. Qwen3-4B reached only .263 even
  with the oracle.
- If the headroom on the bank is below 5 points, `not_found_at_this_scale` is guaranteed. The "selection is not the
  bottleneck" reading would then be an artifact.

**Fix:** before sealing, run a pilot on a separate pilot slice of about 60 dialogues, in neither validation nor
holdout (a `build/` script, like `agentic-coding-small/build/pilot.sh`).
- Arms: `off`, baseline, and a label-derived oracle (≤ 1,024 tokens).
- Measure: oracle − baseline, paired sd, output-failure rate and minutes per 100 items.
- Proceed only if oracle − baseline is at least about 2 × MES. Record the result in DECISIONS.md.

## SHOULD-FIX

**S1. The novelty claim is overstated, and IDEA.md misses the closest prior.**
"What it never searched is the content policy of the reminder" is not accurate:
- Exp 3B ran a label-free selector with supersession (FOCUS-3 `auto`) at 4B on 16 short items
  (`results/memorycode-derived/setup-4b/summary.json`).
- Its `fraction_required` scores: restate_all .099, auto .203, oracle .255. Auto beat restate_all by +10.4 points
  [−7.3, +31.3].
- The experiment was then declared INELIGIBLE on its strict gate (`plan/LEDGER.md` around line 3873).
- C2 found that a parameter-free role rule beat a learned selector.

**Fix:** cite these results. Reframe the novelty as "no *search* over policies in the long/evicted regime". The
hypothesis already has this prior behind it.

**S2. Reminder size is confounded with selection.** The surface may use up to 1,024 tokens, while the baseline uses
256. A win may just mean "restate more" rather than "select better", which would weaken the conclusion that the
stencil is where the headroom sits.

**Fix:** add a second frozen reference, `evicted_1024` (the same newest-first policy with a 1,024-token budget). The
report can then separate size from selection. As a side effect, the power check uses the larger of the references'
item sds, which is the conservative choice.

**S3. The surface cannot implement the baseline reliably.**
- The 4C baseline selects sentences evicted relative to the base-window cut, which is the cut without a reminder
  (see the 4C disclosure about reprovenance).
- `plan()` gets sentences plus `count_tokens`, so the surface would have to reimplement `build_long_prompt`'s
  tokenize-and-retrim loop.

**Fix:** the harness passes, for each sentence, `evicted_at_base_window` and its token count, plus a frozen helper
listed in `agent.visible` for "what is displaced at budget B". Add a test that the baseline surface reproduces
`evicted_mentor_sentences` + `pack_long` byte-for-byte on sample bank items.

**S4. Where the snapshot lands, and label privacy.**
- `git archive <sha> | tar -x` copies the whole tree: `vendor/memorycode` (the `topics.json` regexes and 360
  labeled dialogues), `results/` (stencil's registered pools and the 4C raw outputs) and `data/bench/`.
- RUN mounts only `/data/train` and `/data/public`, so whatever the harness imports must sit there. If the whole
  archive goes there, "topics.json is private" becomes false.
- The import list in IDEA.md is also incomplete. `focus3.py` imports `stencil.focus2`, which imports
  `stencil.stats`.

**Fix:** path-limit the archive.
- RUN-visible: `src/stencil/{__init__,memorycode,focus3,focus2,stats}.py`.
- Private only (EVALUATE): `vendor/memorycode`.
- Also read `data/g0/chat.jsonl` in PREPARE (B3).
- Nothing else is extracted.

**S5. The pivot-disjoint split needs stratifying.**
- Only pivots 6–15 can ever be updated (`generate_template.py`: `instruction_ids_with_updates = range(6, 16)`).
  These are 5 affix families, each with a start and an end pivot.
- 9 families have a single pivot: the two annotation families and the try, assert and docstring families (method
  and function each), plus comment.
- A random split can leave one side with little supersession and a different mix of required families. Comment,
  variable and import are always required.

**Fix:**
- Split the affix pairs across sides (6|11, 7|12, …).
- Spread the multi-pivot families across both sides.
- Record where each singleton family goes.
- Test that the per-split update counts and family mix are similar.

**S6. The timing and budget arithmetic is optimistic.**
- Measured on this machine (`DECISIONS.md:44-45`): vLLM aggregate decode for Qwen3.5-4B was about 160–200 tok/s at
  16–40 streams, though with long contexts.
- For this pack: 4C hit the 512-token cap on 43% of outputs, so a 1,024 cap implies roughly 150–200 k decode tokens,
  plus 1.1 M prefill tokens per run. That is plausibly 10–20 min, not 4–7.
- 60 experiments in 6 h cannot fit even at 7 min per run. Each experiment also needs a proposal (about 1–1.5 min,
  from memory-longmemeval's `results.tsv`) and, when the screen passes, a confirm run. Calibration (3 baseline runs
  plus references) and FINALIZE (4 holdout runs) add more.

**Fix:**
- Measure the time in the B4 pilot.
- Set `--max-num-seqs` to about 64.
- Choose the generation cap (512 as in 4C, or 768 or 1,024) before sealing.
- Either set `max_hours` to about 10–12, or state that you expect about 20–30 experiments.
- Re-derive `run.timeout_s`.

**S7. Drop the auxiliary `llm` from v1.** Its ≤ 4,096 tokens per item cannot hold even the evicted region, since
histories are ≥ 7,168 tokens by design. Using it fully would roughly double run time and trip the 1.5× wall-clock
guard. It also adds budget surface without a coherent use.

**S8. The "28 free real dialogues" secondary metric is empty.**
- They are the short-eligible leftovers: 2–5 sessions, histories at most about 3.7 k tokens. By construction none is
  in the 212-dialogue long set.
- So at W = 3,584 nothing is evicted, and the metric cannot measure the question.

**Fix:** drop it, or score them descriptively at a reduced W (for example 1,536). They are the only real, unregistered
data available, so the reduced-W version is worth keeping.

**S9. Hard-coded `topics.json` strings.** The pivot-disjoint holdout does not stop a surface that embeds MemoryCode's
public instruction list, which IDEA.md itself says the agent probably knows. With verbatim topic text in the bank,
such a surface is effectively a label-derived oracle.

**Fix:** EVALUATE marks the run `invalid` if the surface source contains any ≥ 5-word n-gram from `topics.json`
instruction texts or eval queries, and program.md states this rule. The paraphrasing in B1 also reduces the risk.

## MINOR

- **M1. Differences from 4C.** The generation cap is 1,024 here versus 512 in 4C, and the runtime is vLLM rather
  than transformers. The baseline will therefore not reproduce 4C's numbers, and the "0.43 in both arms" guard
  rationale belongs to 4C's setup. Say so, and pin `Qwen/Qwen3-4B` to revision `1cfa9a7208912126459214e8b04321603b3df60c`
  (the 4C trunk).
- **M2. Seed-pack switch.** "The orchestrator may switch to the deterministic defaults" is not possible within a tag,
  because seeds are sealed. Keep the seed pack and delete that sentence. memory-longmemeval measured sigma = 0 at
  temperature 0, but its answers were only 32 tokens long.
- **M3. Guard name.** The runner's RUN wall-clock measurement is `train_s`. Use that name; PACK-AUTHORING suggests a
  1.3× ratio.
- **M4. Verdict target.** State `verdict: {compare_to: baseline}` explicitly.
- **M5. Rendering details.**
  - Say which oasst2 role becomes the mentor. It decides what the baseline restates.
  - Add a test that every inserted instruction is exactly one `focus3.sentences` span, since the splitter breaks on
    `e.g.` and similar abbreviations.
- **M6. Query-conditioned selection.** Choosing class-side or function-side conventions from the query is
  legitimate. It does exploit the frozen required-family rule, so say so in program.md.
- **M7. Caveat the "15 points".** It rests on n = 16, with interval [+4.0, +27.3], on the research runtime, with the
  oracle clipped to 256 tokens on 8 of the 16 items.
- **M8. Local models.** State whether the surface may load local models from `/hf` (for example an embedder). The
  current rule, "never reads files", is ambiguous on this.

## Statistics summary

MES 0.05 is well powered at 300 holdout items:
- With the 4C paired sd (0.126): 2·SE ≈ 0.015.
- With the sd of oracle − focus on the 4B SETUP items (0.244, computed from the RESULTS-4B per-item table):
  2·SE ≈ 0.028.
- At 250 items with sd 0.244, it is still ≈ 0.031.

The baseline is the right comparison target (4C shipping policy); `off` is the right power reference. The seed pack
guarantees sigma is measured, whether or not vLLM happens to be bit-stable. The remaining statistical risk is not
power. It is whether the bank has any headroom (B4) and whether the effect is really about selection rather than
size (S2).

## Verdict

**IDEA.md is not ready to build from yet.**

The question itself is faithful and sized for arlab, and its cited numbers and pool accounting check out. It follows
directly from stencil-llm's open result: restatement is the strongest lever, and the gap between the oracle and
evicted-restatement is the headroom. MES and the reference setup are statistically sound.

The problem is the data design. As written, the synthetic bank:
- makes selection trivial (B1), with its construction visible to the agent;
- cannot be filled from code-free oasst2 at the stated size (B2);
- would draw on a stencil TRAIN pool (B3);
- has no demonstrated headroom (B4).

Together these would make either verdict uninterpretable.

**What to do:**
1. Revise the Data section: paraphrased conventions and realistic distractors, a trivial-selector ceiling test, a
   chatter-reuse or second-corpus plan with realistic lengths, the `data/g0` exclusion, and a hidden construction.
2. Run the pre-seal pilot for headroom and timing.
3. Fold S2–S7 into the pack spec: the `evicted_1024` reference, per-sentence eviction flags, a path-limited
   snapshot, a stratified split, realistic hours and generation cap, and no auxiliary `llm`.

After that, the pack can be built with the normal §4 workflow and the Astra review.
