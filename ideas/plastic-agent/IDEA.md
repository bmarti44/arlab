# plastic-agent — Stage 1 "FauxOS": explore, consolidate into a per-world LoRA, act without the transcript (v1: redesigned after gate 1)

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
does it close, at ~5% of its prompt tokens?

## Design

**Model.** Qwen3-1.7B (cached snapshot `70d244cc`, bf16, 28 layers), post-trained checkpoint with thinking disabled.
HF transformers 4.56.2 in the NGC PyTorch 25.10 image (the looped-latent pin); no vLLM. Generation uses a frozen engine
that computes the shared prefix (system prompt, plus the transcript for the ICL arm) once per world and copies its KV
into a static cache per batch; a CPU test checks it token-for-token against uncached greedy decoding.

**FauxOS (frozen/prepare/fauxos.py; PREPARE and EVALUATE only, never mounted into RUN).** A deterministic pure-Python
world generated from one seed: 4 kinds, 4 places, 6 tags, a weight unit (display and base, conversion constant C in
2..9), three error codes (locked / missing / bad argument), 30–60 objects with hidden fields (kind, place, weight,
created order, locked, archived, tag). 11 tools: `inspect` plus 10 of 15 task operations (archive, delete, restore,
move, lock, unlock, retag, clone, swap, count, weigh, extreme, newest, checksum, find_tag), sampled per world. Every
tool gets a pseudo-word name whose verb is a pseudo-word (50%), a truthful English verb (15%) or a misleading one (35%:
an `_open` that locks, a `_stash` that deletes), a random positional argument order and per-op variants (counts with or
without locked items, totals in either unit, heaviest vs lightest, newest vs oldest, checksum constants). Every tool
prints what it did ("archived #412", "copied #881 as #1000"), so each tool's semantics are visible in its log lines.

**Exploration (frozen at PREPARE).** A scripted explorer (not the base model: PREPARE is CPU-only, and a scripted policy
makes the experience identical for every arm and independent of the model) makes 120 calls per world: one sensible call
to every tool, a pass that triggers each error code, two more passes over every tool, then a round-robin of sensible
calls (85%) with occasional inspect follow-ups and random probes (15%: random ids, reversed argument order, wrong
arity). Every tool appears at least 7 times with a successful call. The transcript is exactly the list of `{call, obs}`
pairs, 2.5–2.8k tokens rendered. Tests replay every call against the simulator and require the same observations, so
the transcript contains nothing but observable facts. Surface-driven exploration is v2.

**Held-out tasks.** Every task is ONE call of one of the world's task operations, from 15 templates, starting from the
state the exploration ended in: mutations ("Archive item 412.", "Move item 17 to <place>.", "Tag item 9 with \"x\".",
"Swap the places of items 3 and 8.") and questions ("What is the tally checksum of <place>?", heaviest/lightest,
newest/oldest, totals in a named unit, tags, counts). Gold = the outcome of the reference call on the simulator.
Rejected: duplicate goals, mutation tasks whose exact reference call appears in the transcript, and state tasks that
change nothing. The model answers with one call (`name(arg, ...)`, positional literals; the format is forgiving: the
first line that starts like a call is the program, a leading `> ` is dropped, prose, fences, predicted outputs and
further calls are ignored, a bare word is a string, a digit string is an int); on a syntax or execution error it gets
one retry showing the error. Success = no error, exact final state (state tasks; collateral changes fail) and exactly
the gold value in the output (answer tasks). No LLM judge. Splits (disjoint seed ranges): validation 4 worlds x 60 tasks
= 240 items; holdout 8 worlds x 80 tasks = 640 items (FINALIZE only); each split has its own guard world (100 tasks),
GSM8K items and text rows for the forgetting battery (disjoint between validation and holdout).

**What RUN sees.** Only `public/order.json` + `public/worlds/<id>.json` (world id, sorted tool names, transcript; the
harness loads one world at a time), `public/twins/<id>.json` (the twin control worlds, same format) and OASST2 replay
rows. Goals, answers, specs, states, the simulator and the
battery items (copied from GSM8K / OASST2 at PREPARE) are in `private/` (EVALUATE only); no RUN-side code references a
dataset under `/hf`. The arm is chosen by frozen code: the sha256 of `/work/adapt.py` against the frozen references
(`common.arm_of`), never a constant in the surface.

**Isolation (as built, v1 rounds 2–3).** `harness.py` is a trusted supervisor that owns the model and never imports the
surface. Per world it starts a FRESH child (`child.py`, `python -s -B`, minimal env, CUDA hidden, its own session) that
locks itself down with Landlock (the latent-arch sandbox): readable only the Python install, `/usr /lib /etc /opt /proc
/sys`, `/frozen` and `/work`; writable only a fresh per-world scratch dir (deleted afterwards) and `/dev/null`; no GPU,
no TCP, no signals or ptrace-style access outside the sandbox. A self-check (every data file RUN can see, the model
snapshot and `/proc/<parent>/mem` must not open; `/tmp` must not be writable) must pass or the run crashes. The child
receives only that world's transcript and tool names over its stdin, disables every Python-level thread/process start,
imports `adapt.py` and calls `adapt()` with RPC stubs (`surface_api.py`). Every generate / teacher / train request is
validated, executed and metered by the supervisor (`engine.Gen`, `trainer.Trainer`, one `Budget` per world, CLOCK_MONOTONIC
from "go", so the import counts for every world); the child cannot pass `budget=None`, touch a counter or move the
deadline, and never holds adapter weights or teacher log-probs: they stay in the supervisor under immutable integer
ids, and `adapt()` returns an id. At deadline + 15 s the child's process group is SIGKILLed (the last adapter trained is
kept). On return the supervisor checks the child from outside (OS threads beyond its startup set, any descendant, any
orphan re-parented to the subreaper supervisor) and records each as a violation (the evaluator invalidates the run),
then kills the process group and every descendant before it saves ITS copy of the adapter (a data-only safetensors
file that the evaluator re-validates: names, shapes, finiteness). Nothing the surface starts can run after `adapt()`
returns (SIGKILL: no atexit), and no state carries from one world to the next. The base weights are sha256-hashed at
load and after every world.

**Surface.** `surface/adapt.py`: `adapt(transcript, tool_names, gen, train) -> AdapterRef | None`, within a frozen per-world
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
| (e) placebo / twin control | world i scored with an adapter trained on world i's TWIN | frozen reference `ref_placebo` (baseline LoRA on the twin's transcript, scored on world i) **and** every candidate run: the same surface adapts to each twin too, `twin_success` = world i scored with that adapter, `world_specific_gain` = (d) − that, guarded by `world_specific_share` |

Only d − e measures environment-specific knowledge: TTT gains on ARC and in GTTA are partly format learning, and a
LoRA from another world teaches the program format without the right facts (TTT-DEEP-DIVE §4).

**Twin (control) worlds (round 3).** Round 2 used another world of the split as the control, but that changes the tool
NAMES too: a names-only surface (fixed verb-family priors, transcript ignored) scored 30–37% (review: 33% / 44%) while
donor adapters emit names that do not exist (~0%), so name priors looked world-specific. Now PREPARE builds, for every
validation / holdout world, a twin (`fauxos.gen_twin`, own seed range): the SAME tool names, operation set (so the same
task templates and wording) and argument vocabulary (kinds, places, tags, units), but a derangement of which name does
what (every name maps to a different operation), fresh argument orders, variants, objects, error codes and unit
conversion, and its own transcript from the same frozen explorer (`public/twins/<id>.json`). For a candidate, RUN runs a
second `adapt()` pass per world on the twin (same sandbox, same budget, the child cannot tell the passes apart except
by the transcript), and EVALUATE scores that twin adapter on the TARGET world's tasks. Name / vocabulary / format
priors adapt identically to both and cancel in `world_specific_gain`; only what is learned from the transcript differs.
In memory, on the real simulator and scorer: the names-only policy gets share 0 (validation 0.30 vs 0.30), a policy
that reads operations and argument orders from the transcript gets 1.00 on its world and 0.00 with the twin's (share 1).

**Metric.** `success` = mean task success over the split's worlds (primary; `items` = one 0/1 per task, so the SE is
paired) for every arm. Also reported: `twin_success`, `world_specific_gain`, `world_specific_share` (candidates),
`none_success` and `icl_success` (from the evaluator's cache once the references ran), `gap_closure` = (d − a) / (b − a),
per-template `s_*`, retry and syntax-error rates, adapt time and tokens, and the inference token accounting per task
(context positions of each model call, the shared system prefix included in each): `initial_context_tokens` (first
call), `retry_prompt_tokens` (the retry call's full context; 0 without a retry), `prefill_tokens` = their sum,
`decode_tokens`, `inference_tokens` = prefill + decode.

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
RESEARCH.md §7, with ARC-Easy replaced by more GSM8K because ARC-Easy is not cached offline); `prefill_tokens <= 1000`
(every prompt position processed per task, the retry call included: the surface arms use ~140 on the first call and
<= ~400 with a retry, the ICL arm ~2.6k first call, ~3.2k with retries); `world_specific_share >= 0.5`
(= (success − twin_success) / (success − none_success), 1.0 when success − none_success < 0.025, so a run that
gains nothing passes vacuously; a gain that is mostly name-prior / format / task-family learning, i.e. that the same
surface also produces from the twin's transcript, fails it); `peak_mem_gb <= 60`; budget `adapt_s_max <= 210` (the
longest `adapt()` pass, target or twin, whose clock starts before the surface is imported); a non-finite loss or gradient in any `train()`
call (sticky across calls), an adapter with non-finite or wrongly-shaped tensors, a failed sandbox self-check or a
thread / process left by the surface makes a run invalid.
Budgets count processed positions (padded prompt blocks, every decode step of every batch row, padded training
batches, replay rows); a generation batch reserves its worst case before it runs, so the caps are never exceeded. OASST2 `text_nll_ratio` is reported, not guarded. There is no `train_s` ratio guard: the budget
itself is wall clock, and the naive baseline uses only a fraction of it.

**Data-prep gate (GPU pilot, `build/pilot.sh gate <dir>`).** Before any campaign: no-adaptation success <= 0.15 and
ICL − none >= 0.15 on validation, else NO-GO: retune `fauxos.N_OPS / TEMPLATES / VERB_MIX` and
`prepare.N_EXPLORE` (a new data hash) and re-run the gate. The pilot then runs the baseline and placebo arms for
timing.

## Constants to calibrate in the GPU pilot (none measured yet; everything so far ran on the CPU with a tiny model)
- Gate: `N_OPS`, `TEMPLATES` mix, `VERB_MIX`, `N_EXPLORE` (120 calls ≈ 2.6k tokens) until the gate says GO.
- Per-world budget `--adapt-seconds 180 --gen-tokens 400000 --train-tokens 1500000` and `budget.limit` (= adapt + 30):
  the naive baseline should take ~30–60 s per world; a rich surface (teacher logits over 2.6k-token contexts) should fit.
- `run.timeout_s` (holdout: 8 worlds) and `evaluate.timeout_s` (ICL reference and the first base-cache fill).
- `battery_drop` threshold: 2 points on 300 items is about 1 SE of noise, so a harmless adapter fails it ~15% of the
  time. If the baseline's own drop is noisy, set the threshold to max(0.02, 2·SE of the measured paired difference).
- `peak_mem_gb` (static KV caches are capped at 262k token slots, ~29 GB) and the campaign limits (~19 min per run).
- Baseline hyper-parameters (8 epochs, r16, lr 2e-4): check that its training loss actually falls; an under-trained
  baseline would make the MES too easy.

## Gate history
- **Gate 1 (GPU, v0 world): NO-GO.** Validation, 240 items: none 0.067, icl 0.121 (+0.054). The ICL arm had retry rate
  0.58 and syntax errors 0.25; every 2-call and bulk template was 0/0. Causes:
  - A 12k-token log of 500 calls over 26–28 tools (with decoys and silent `ok` mutations) was too much for 1.7B to use.
  - The strict multi-line program format failed on log-style output: `> call`, a predicted output line, then more calls.
  - Unquoted string arguments were syntax errors.
  - `delete` was 0.32 → 0 because the explorer showed deletes mostly as errors on random ids and printed `ok`.
- **Redesign (v1, CPU only).**
  - 11 tools, and every tool reports its effect.
  - 120-call transcripts (~2.6k tokens), with every tool shown at least 7 times.
  - Single-call tasks only.
  - A forgiving one-call program format.
  - Fewer truthful verbs (15%).
  - Prefill guard 3000 → 1000.
- **CPU estimate of the v1 gate** (Qwen3-1.7B fp32 on CPU, first attempt only, no retry, the first tasks of each world):
  - none 5/96 = 0.05 over all 4 validation worlds.
  - icl 15/24 = 0.62 on v0 and v3 of the final data, and 36/48 = 0.75 on all 4 worlds of the previous v1 iteration.
  - ICL failures are what the adapter must learn: inverted lock/unlock names followed over the log, and permuted
    argument order. Expected GPU gate: none ≈ 0.05–0.10, icl ≈ 0.6–0.8, so GO with a wide margin.

## Honest expectations (RESEARCH.md §7, TTT-DEEP-DIVE.md §4)
With the v1 world, ICL − none should be 50–70 points. The naive baseline should close under 20% of that gap.
An ARC-style surface (hindsight relabeling + dynamics pairs + augmentation, teacher KL as a supplement) plausibly closes
30–60% (+5 to +15 over none); TTT-DEEP-DIVE puts the chance that it clears MES 0.06 over the naive baseline at ~35–45%.
Beating ICL at 1.7B is unlikely. The prefill saving (~2.5k tokens per call) holds
either way. The largest single risk is that the relabeled demos teach the format but not the inverted semantics; the
placebo arm, `world_specific_gain` and the `world_specific_share` guard are there to catch that.

## Caveats and open risks (for REVIEW.md)
1. **Deviations from the brief.** The explorer is scripted, not the base model (PREPARE has no GPU). No vLLM: `gen` is
   the HF engine, so self-study generation with the transcript in context is slower than the brief's estimate. ARC-Easy
   is not cached; the battery is GSM8K + the guard world. The brief's "4 instances x 40 tasks per seed" became fixed
   validation / holdout items, because arlab item packs pair items across seeds. Tasks start from the post-exploration
   state. The perturbed-generator holdout (RESEARCH risk 3) is not built (v2). After gate 1 the world was simplified
   (v1: 11 tools, 120-call logs, single-call tasks, forgiving format): Stage 1 now measures consolidation of renamed and
   inverted tool semantics and argument orders, not multi-step planning (2-call tasks were 0/0 even with ICL).
2. **World clustering.** The SE treats tasks as independent; they are clustered in 8 holdout worlds, so a verdict is
   about these worlds. `s_*` per template and per-world numbers in the logs show heterogeneity.
3. **Weak teacher.** 1.7B ICL may be too weak to distil (SDFT failed at 3B). The stretch model is Qwen3.5-4B; it needs a
   new tag and a check that the frozen LoRA and cache code handle its hybrid layers.
4. **Surface isolation (round 2: fixed, see "Isolation").** adapt() no longer runs in the harness process: it runs in a
   fresh Landlock-sandboxed child per world and reaches the model only by metered RPC; threads / processes it leaves
   are caught from outside and invalidate the run. What remains: a thread started and ended between two RPCs without
   the Python-level API (ctypes) is not seen, but it can only compute on the CPU inside the metered wall clock, with
   no model, data or writable path, and is killed with the process group when adapt() returns.
5. **Generator-family priors.** A surface could learn "semantics are often inverted" rather than this world's facts.
   That is arguably the legitimate general skill; the placebo arm and the `world_specific_share` guard (>= 0.5 of the
   gain over none must need the right world's transcript) bound how much of a kept gain may be of that kind. The guard
   is noisy: on 240 validation items the paired SE of success − twin_success is ~0.02–0.03, so a gain of 0.06
   passes only if its world-specific part is >= 0.03 (about 1 SE); a genuinely world-specific candidate can fail it by
   chance, and it is checked on the screen seed and both confirm seeds.
   **Twins are distinguishable** (residual, not fixable without new worlds): target names were generated conditioned
   on the target's semantics (15% truthful, 35% misleading English verbs), so in a target 1–10 of 11 tool verbs match
   the verb tables for what the transcript shows the tool doing, in a twin 0–2 (83% of twins: 0; 400 simulated pairs).
   A surface that detects twins that way and deliberately adapts worse on them would get share ~1 for gains that
   come from name priors. program.md forbids it and it would be visible in the diff; making twins indistinguishable
   needs worlds whose names do not depend on their semantics (v2, new worlds).
6. **Wall-clock budget on a shared GPU.** Slow but uncontended periods shrink what fits in 180 s; token caps are the
   machine-independent part of the budget.
7. **Claim scope: unseen worlds, not unseen task types.** Worlds (seeds, tool names, verbs, argument orders, units,
   objects) are disjoint between validation and holdout, but the 15 goal templates and their wording
   (`fauxos.TEMPLATES` / `make_tasks`), the 15 operation types and the output grammar are shared across splits. A kept
   result therefore says the surface consolidates never-seen WORLDS of known task families; it says nothing about
   generalization to new templates, phrasings or operation types (that would need template / paraphrase families
   partitioned by split: v2 with the perturbed-generator holdout).
8. **HF cache in campaigns.** arlab mounts the whole HF cache read-only at `/hf` in RUN and EVALUATE (runner level,
   `campaign.trial()`; `pack.yaml` cannot narrow it; only `build/pilot.sh` mounts just the model). So the raw GSM8K test
   split and OASST2 dump from which PREPARE copies the private GSM8K battery, the text-NLL rows and the replay rows are
   on disk in RUN. The pack no longer depends on them being unreadable to the surface: adapt.py runs under Landlock with
   `/hf` (and every data file) unreadable, and a test opens exactly those two source files (and the model snapshot)
   from the sandboxed surface and requires the open to fail. The trusted supervisor reads only the model snapshot from
   `/hf`. The research agent's container does not mount `/hf` (`arlab/agent.py`: only its view, `/out` and CODEX_HOME). Residual: the base model
   may have seen GSM8K test items in pretraining, which affects base and adapted models alike (battery_drop is a
   difference).
9. **Next (v2):** sequential A→B adaptation in one LoRA with a re-test on A; surface-controlled exploration at a matched
   token budget; a KV-prefix (Cartridge) arm; stage 2 meta-training of a "TTT-able" base if stage 1 clears the MES.
