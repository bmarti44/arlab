# plastic-agent — Stage 1 "FauxOS": explore, consolidate into a per-world LoRA, act without the transcript (v0)

**The owner's idea.** An agent that meets an unfamiliar environment should get better at it the way people do: by
writing the experience into its own parameters, not by carrying the whole session transcript in its context forever.
RESEARCH.md frames it as "an agent adapts its parameters (full weights, a LoRA, or a trained KV prefix) online, from its
own interaction with a new environment. It then does better there *without* the transcript in context."

**The driving analogy (RESEARCH.md §4).** Humans are not zero-shot either: drivers new to a place are over-involved in
some crash types. They transfer because a strong general model is combined with fast local learning, and Complementary
Learning Systems theory describes the mechanism: the hippocampus stores episodes quickly and replay slowly consolidates
them into cortex. For an LLM, pretraining is the general model and the KV cache / in-context learning is the
hippocampus (fast, flexible, lost at session end, paid for in prefill on every call); "the missing piece is
consolidation". FauxOS makes the world a place where "the semantics are the 'drive on the left': pretraining priors are
wrong, not just missing", and the forgetting guard is "the 'can it still drive in San Jose' check".
Background and sources: `RESEARCH.md` (§7 is this experiment) and `TTT-DEEP-DIVE.md` (§4: ranked recommendation, Stage 1,
the placebo arm, and the rule that relabeling may use only what the transcript shows).

**Question at this scale.** After a frozen exploration of a never-seen tool world, can Qwen3-1.7B consolidate that
experience into a per-world LoRA that solves held-out tasks in that world **with no transcript in context**, better than
naive next-token LoRA on the transcript by at least the MES; and how much of the gap to the transcript-in-context arm
does it close, at ~2% of its prompt tokens?

## Design

**Model.** Qwen3-1.7B (cached snapshot `70d244cc`, bf16, 28 layers), post-trained checkpoint with thinking disabled.
HF transformers 4.56.2 in the NGC PyTorch 25.10 image (the looped-latent pin); no vLLM. Generation uses a frozen engine
that computes the shared prefix (system prompt, plus the transcript for the ICL arm) once per world and copies its KV
into a static cache per batch; a CPU test checks it token-for-token against uncached greedy decoding.

**FauxOS (frozen/prepare/fauxos.py; PREPARE and EVALUATE only, never mounted into RUN).** A deterministic pure-Python
world generated from one seed: 4 kinds, 4 places, 6 tags, two weight units with a conversion constant C in 2..9, three
error codes (locked / missing / bad argument), 30–60 objects with hidden fields (kind, place, weight, created order,
locked, archived, tag). 25 operations + 1–3 decoys (a second list/count/extreme/newest/checksum/weigh tool with another
variant) = 26–28 tools. Every tool gets a pseudo-word name whose verb is a pseudo-word (45%), a truthful English verb
(25%) or a misleading one (30%: an `_open` that locks, a `_purge` that archives), a random positional argument order and
per-op variants (list order newest/oldest/id and whether locked items are skipped, counts with or without locked items,
weights in either unit, bulk thresholds strict or not and in either unit, heaviest vs lightest, newest vs oldest,
checksum constants, lock that toggles). Half of the mutating tools print their effect; the others print only `ok`, so
their semantics must be inferred from a later `inspect`.

**Exploration (frozen at PREPARE).** A scripted explorer (not the base model: PREPARE is CPU-only, and a scripted policy
makes the experience identical for every arm and independent of the model) makes 500 calls per world: an overview,
one sensible call to every tool, a pass that triggers each error code, then 70% sensible calls with inspect follow-ups
after mutations and 30% random probes (random ids, reversed argument order, wrong arity). The transcript is exactly the
list of `{call, obs}` pairs, ~11–13k tokens rendered. Tests replay every call against the simulator and require the same
observations, so the transcript contains nothing but observable facts. Surface-driven exploration is v2.

**Held-out tasks.** Goals from 22 templates, starting from the state the exploration ended in: single mutations
("Archive item 412."), questions ("What is the tally checksum of <place>?", heaviest/lightest, newest/oldest, totals in a
named unit, tags, counts), bulk operations with unit conversion ("Archive every unlocked <KIND> item that weighs at least
3 <unit>."), and 2-call compositions ("Unlock item 305, then permanently delete it.", "Archive item 17, then report the
tally checksum of <place>."). Gold = the outcome of a reference program on the simulator. Rejected: duplicate goals,
mutation tasks whose whole reference program appears verbatim in the transcript, and state tasks that change nothing.
The model writes a program (one call per line, positional int / "string" literals, <= 12 lines, the last output is the
answer, `answer(x)` outputs x); on a syntax or execution error it gets one retry showing the failing line and its output.
Success = no error, exact final state (state tasks; collateral changes fail) and exactly the gold value in the last
output (answer tasks). No LLM judge. Splits (disjoint seed ranges): validation 4 worlds x 60 tasks = 240 items; holdout
8 worlds x 80 tasks = 640 items (FINALIZE only); each split has its own guard world (100 tasks), GSM8K items and
text rows for the forgetting battery (disjoint between validation and holdout).

**What RUN sees.** Only `public/order.json` + `public/worlds/<id>.json` (world id, sorted tool names, transcript; the
harness loads one world at a time) and OASST2 replay rows. Goals, answers, specs, states, the simulator and the
battery items (copied from GSM8K / OASST2 at PREPARE) are in `private/` (EVALUATE only); no RUN-side code references a
dataset under `/hf`. The harness gives `adapt()` only the world being adapted and checks, with a sha256 over every
base parameter and buffer, that the base weights are byte-identical (and no LoRA module is left) at load, between
worlds and at the end. The arm is chosen by frozen code: the sha256 of `/work/adapt.py` against the frozen references
(`common.arm_of`), never a constant in the surface.

**Surface.** `surface/adapt.py`: `adapt(transcript, tool_names, gen, train) -> adapter | None`, within a frozen per-world
budget (180 s wall clock, 400k gen tokens, 1.5M train tokens; calibrate). `gen` = budgeted base-model generation and
teacher top-k log-probs in the evaluation's chat format, with or without the transcript in context. `train` = the frozen
LoRA loop (surface chooses examples: raw text or prompt/completion pairs with optional teacher top-k; loss weights for
CE, teacher KL and KL-to-base on replay rows; rank <= 64, targets, layers, lr, epochs, batch). **Baseline surface = naive
next-token LoRA on the raw transcript** (r16, all projections, 8 epochs, lr 2e-4): the SEAL / CodeUpdateArena control.

**Arms** (one prompt format for all: system prompt with the tool names only, no docs; then the goal):

| Arm | What it is | How it is run |
|---|---|---|
| (a) none | no adaptation | frozen reference `ref_none` |
| (b) icl | the whole transcript in the system prompt, no adapter | frozen reference `ref_icl` (fails the prefill guard by design; never a candidate) |
| (c) guard | forgetting battery with each adapter | every run: guard `battery_drop` |
| (d) surface | the per-world adapter; baseline = naive next-token LoRA | the campaign's candidates; the verdict compares (d) with the baseline |
| (e) placebo | world i scored with the adapter trained on world i+1 | frozen reference `ref_placebo` (baseline LoRA; the harness hands world i the transcript of world i+1, the evaluator scores world i as for every arm) **and** every candidate run reports `cross_world_success` (its adapter j on world j−1) and `specific_gain` = (d) − that |

Only d − e measures environment-specific knowledge: TTT gains on ARC and in GTTA are partly format learning, and a
LoRA from another world teaches the program format without the right facts (TTT-DEEP-DIVE §4).

**Metric.** `success` = mean task success over the split's worlds (primary; `items` = one 0/1 per task, so the SE is
paired) for every arm. Also reported: `cross_world_success`, `specific_gain`, `none_success` and `icl_success` (from the evaluator's cache
once the references ran), `gap_closure` = (d − a) / (b − a), `prefill_tokens`, per-template `s_*`, retry and syntax-error
rates, adapt time and tokens.

**MES (fixed before calibration): 0.06** (6 points over the naive-LoRA baseline; RESEARCH.md §7). Item pack, holdout
640 items, 3 holdout seeds: SE = sqrt(sd²/640 + 2σ²/3), where σ is the SD of the 5 calibration runs.
- With the runner's default sd = sqrt(0.2): σ = 0 gives SE 0.0177 (2.5·SE = 0.044); σ = 0.015 gives SE 0.0215
  (2.5·SE = 0.054). Both meet the recommended MES >= 2.5·SE, and the pack is powered (2·SE <= 0.06) for σ <= 0.030.
- Because the pack has references, the runner's power check uses the largest measured reference-vs-baseline item sd,
  which will be the ICL arm's. If ICL beats the baseline by ~35 points on ~40% discordant items, sd ≈ 0.55: σ = 0 gives
  SE 0.0217 (2·SE 0.043), σ = 0.015 gives SE 0.0250 (2·SE 0.050); powered iff σ <= 0.025 (σ <= 0.022 at sd 0.60).
  This is why the holdout has 640 items. 2.5·SE then sits at 0.054–0.062, just around the MES.
- If CALIBRATE reports underpowered, add holdout worlds or tasks under a new tag; never change the MES.
Seeds: calibration [1–5], screen 1, confirm [2, 3], holdout [101–103] (LoRA init and data order vary with the seed; the
evaluation is greedy and deterministic).

**Guards.** `battery_drop <= 0.02` (base minus adapted accuracy on 300 items: 200 GSM8K test problems and 100 tasks of
the guard world with its transcript in context, each item scored with one of the run's adapters, round-robin; per
RESEARCH.md §7, with ARC-Easy replaced by more GSM8K because ARC-Easy is not cached offline); `prefill_tokens <= 3000`
(the surface arms use ~220 prompt tokens, the ICL arm ~12k); `peak_mem_gb <= 60`; budget `adapt_s_max <= 210`
(the longest per-world `adapt()`, whose clock starts before the surface is imported); a non-finite loss or gradient in
any `train()` call (sticky across calls) or an adapter with non-finite or wrongly-shaped tensors makes a run invalid.
Budgets count processed positions (padded prompt blocks, every decode step of every batch row, padded training
batches, replay rows); a generation batch reserves its worst case before it runs, so the caps are never exceeded. OASST2 `text_nll_ratio` is reported, not guarded. There is no `train_s` ratio guard: the budget
itself is wall clock, and the naive baseline uses only a fraction of it.

**Data-prep gate (GPU pilot, `build/pilot.sh gate <dir>`).** Before any campaign: no-adaptation success <= 0.15 and
ICL − none >= 0.15 on validation, else NO-GO: retune `fauxos.TEMPLATES / VERB_MIX / VERBOSE_P` and
`prepare.N_EXPLORE` (a new data hash) and re-run the gate. The pilot then runs the baseline and placebo arms for
timing.

## Constants to calibrate in the GPU pilot (none measured yet; everything so far ran on the CPU with a tiny model)
- Gate: `TEMPLATES` mix, `VERB_MIX`, `VERBOSE_P`, `N_EXPLORE` (500 calls ≈ 12k tokens) until the gate says GO.
- Per-world budget `--adapt-seconds 180 --gen-tokens 400000 --train-tokens 1500000` and `budget.limit` (= adapt + 30):
  the naive baseline should take ~30–60 s per world; a rich surface (teacher logits over 12k-token contexts) should fit.
- `run.timeout_s` (holdout: 8 worlds) and `evaluate.timeout_s` (ICL reference and the first base-cache fill).
- `battery_drop` threshold: 2 points on 300 items is about 1 SE of noise, so a harmless adapter fails it ~15% of the
  time. If the baseline's own drop is noisy, set the threshold to max(0.02, 2·SE of the measured paired difference).
- `peak_mem_gb` (static KV caches are capped at 262k token slots, ~29 GB) and the campaign limits (~19 min per run).
- Baseline hyper-parameters (8 epochs, r16, lr 2e-4): check that its training loss actually falls; an under-trained
  baseline would make the MES too easy.

## Honest expectations (RESEARCH.md §7, TTT-DEEP-DIVE.md §4)
If the world is tuned right, ICL − none should be 20–40 points. The naive baseline should close under 20% of that gap.
An ARC-style surface (hindsight relabeling + dynamics pairs + augmentation, teacher KL as a supplement) plausibly closes
30–60% (+5 to +15 over none); TTT-DEEP-DIVE puts the chance that it clears MES 0.06 over the naive baseline at ~35–45%.
Beating ICL at 1.7B is unlikely except where 12k+-token ICL degrades. The prefill saving (~12k tokens per call) holds
either way. The largest single risk is that the relabeled demos teach the format but not the inverted semantics; the
placebo arm and `specific_gain` are there to catch that.

## Caveats and open risks (for REVIEW.md)
1. **Deviations from the brief.** The explorer is scripted, not the base model (PREPARE has no GPU). No vLLM: `gen` is
   the HF engine, so self-study generation with the transcript in context is slower than the brief's estimate. ARC-Easy
   is not cached; the battery is GSM8K + the guard world. The brief's "4 instances x 40 tasks per seed" became fixed
   validation / holdout items, because arlab item packs pair items across seeds. Tasks start from the post-exploration
   state. The perturbed-generator holdout (RESEARCH risk 3) is not built (v2).
2. **World clustering.** The SE treats tasks as independent; they are clustered in 8 holdout worlds, so a verdict is
   about these worlds. `s_*` per template and per-world numbers in the logs show heterogeneity.
3. **Weak teacher.** 1.7B ICL may be too weak to distil (SDFT failed at 3B). The stretch model is Qwen3.5-4B; it needs a
   new tag and a check that the frozen LoRA and cache code handle its hybrid layers.
4. **In-process surface.** adapt() runs in the harness process: it could read the model through `gen` internals or burn
   time. The harness catches base-weight edits (sha256 of the weights at load, between worlds and at the end), saves
   the trainer's private copy of each adapter, the budget is enforced by deadline and reserved token caps, and
   program.md forbids the rest (good-faith proposer). A campaign mounts the whole HF cache read-only into RUN, so the
   raw GSM8K test set is readable from adapt.py by file I/O; no frozen RUN code references it (tested) and program.md
   forbids file reads. See REVIEW.md. No goal, answer, state or simulator is reachable
   from RUN, and the generator code is not mounted there.
5. **Generator-family priors.** A surface could learn "semantics are often inverted" rather than this world's facts.
   That is arguably the legitimate general skill; the placebo arm measures how much of a gain is world-specific.
6. **Wall-clock budget on a shared GPU.** Slow but uncontended periods shrink what fits in 180 s; token caps are the
   machine-independent part of the budget.
7. **Next (v2):** sequential A→B adaptation in one LoRA with a re-test on A; surface-controlled exploration at a matched
   token budget; a KV-prefix (Cartridge) arm; stage 2 meta-training of a "TTT-able" base if stage 1 clears the MES.
