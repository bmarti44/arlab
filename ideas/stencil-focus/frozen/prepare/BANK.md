# BANK.md — how the stencil-focus bank is built (NOT agent-visible)

`frozen/prepare/` is mounted only into PREPARE (at `/prepare`) and into the pack tests (via `/pack`); the agent sees
only `program.md` and `frozen/run/window.py`. Code: `frozen/prepare/bank.py` (sampling/rendering) and
`frozen/run/prepare.py` (sources, splits, outputs). Everything is deterministic (string-seeded `random.Random`) and
LLM-free at PREPARE time.

## Sources (all pinned)
| what | where | use |
|---|---|---|
| stencil-llm `87b72a88` (GPL-2.0) | `frozen/prepare/stencil-<sha>.tar.gz` = `build/snapshot.sh`: `git -C ~/stencil-llm archive <sha> src/stencil/{__init__,memorycode,focus3,focus2,stats}.py vendor/memorycode data/g0/chat.jsonl \| gzip -n` | PREPARE checks the sha256 and the commit id git writes into the tar header, extracts to `/tmp`. The 5 modules go to every `public/stencil/src` (RUN + EVALUATE import them); `vendor/memorycode/{code,topics.json,LICENSE,README.md}` go to `private/` only; `data/g0/chat.jsonl` is read for the exclusion and not kept. |
| MemoryCode `topics.json` (Apache-2.0) | in the snapshot | 51 pivots (text/regex variants, eval queries), 30 filler instructions, 19 contexts |
| `Qwen/Qwen3-4B` `1cfa9a72…` | `/hf` (snapshot_download) | the served model; its `tokenizer.json` counts every token |
| `HuggingFaceH4/ultrachat_200k` `8049631c…` `test_sft` (MIT) | `/hf` | chatter |
| `OpenAssistant/oasst2` `179dd21f…` trees (Apache-2.0) | `/hf` | chatter, minus the 29 g0 trees |

Why the tarball is made on the host: PREPARE runs in a container that cannot see `~/stencil-llm`; the archive is
exactly the path-limited command in IDEA.md, and PREPARE refuses any other content.

## Chatter
Thread = one ultrachat conversation, or the best-ranked English branch of an oasst2 tree (prompt, then repeatedly
the non-deleted English reply with the lowest `rank`). oasst2 trees whose root id is one of stencil's 29 g0 trees
(`data/g0/chat.jsonl` ids `<root>:<leaf>`) are dropped. A thread is kept only if it starts with the user, contains an
assistant turn, and no turn contains a code fence (```` ``` ````), "as an AI" or more than 4,000 characters.
Whitespace is flattened (one line per turn). Programming vocabulary is NOT scrubbed. Roles: user -> mentee,
assistant -> mentor. The pool is shuffled (seed "chatter-pool") and cut 45% / 45% / 10% into validation / holdout /
pilot pools; a thread is consumed once (the unused rest of a thread is discarded, never reused).

## Pivot split (IDEA.md: disjoint, stratified)
| side | used by | pivots |
|---|---|---|
| A | validation (+ half the pilot) | 0, 6, 45, 12, 2, 8, 47, 14, 4, 10, 49, 16, 18, 21, 22, 24, 25, 27, 29, 31, 35, 39, 42, 34, 38, 43 |
| B | holdout (+ half the pilot) | 50, 11, 1, 7, 46, 13, 3, 9, 48, 15, 5, 17, 19, 20, 23, 26, 28, 30, 33, 37, 40, 32, 36, 41, 44 |

- Affix pairs across sides, alternating start/end so each side has one updatable pivot per object:
  A = {6 fn-start, 8 method-start, 10 arg-start, 12 var-end, 14 attr-end}, B = {11, 13, 15, 7, 9}.
- Multi-pivot families spread: class 0|50, chx 2,4 | 1,3,5, digit 45,47,49 | 46,48, imports 25,27,29 | 26,28,30,
  function decorators 31,35,39,42 | 33,37,40, method decorators 34,38 | 32,36,41, class decorators 43 | 44.
- Singleton families: A = function annotation (16), method try (18), function assert (21), method docstring (22),
  comment (24); B = method annotation (17), function try (19), method assert (20), function docstring (23).
  Comment exists on side A only (a singleton cannot be on both sides).
- Tests check: similar update counts, live-set sizes, class/function query share, history lengths and the share
  of always-required families (variable, import) between the splits.

## Dialogue template (port of MemoryCode `generate_template.py`, long-dialogue settings of `generate_dataset.sh`)
- Sessions n ~ U{12..30}; lengths short/medium/long (weights 1/2/3); instruction sessions = int(n·U(0.5, 0.7));
  "instruction-only" sessions up to n/5 (here: they get no filler instruction; every session has chatter);
  1-2 instruction events per instruction session; update rate U(0.3, 0.7).
- `sample_instructions` is ported line by line over the side's pivots (per-type candidate draw, insertion vs
  update, updatable pivots 6-15 only, `shuffle_updates` variants, a pivot leaves once its variants are used). One
  guard: when nothing can be introduced or updated the event is skipped (upstream would crash).
- Filler instructions (30 ids, 101-130): per session with P = 1 - U(0.5, 0.7) (upstream `filler_instruction_rate`
  is the topic-filler rate); update with P = U(0.5, 0.8); variants popped last-first as upstream.
- The item is the final session s = n-1 (resampled until its `history_regex` is non-empty and some earlier session
  has an instruction). Query = seeded choice among the live pivots' eval queries. `history_regex` = latest regex per
  live pivot (upstream order); `required` / `structure` from stencil's `required_families` / `required_structure`.

## Lengths
Target history (sessions 0..s-1) ~ U(6k, 20k) estimated tokens (4.2 chars/token), split over sessions by length
weight; each session takes whole user/assistant exchanges from fresh threads until its target, ending on a mentor
turn. Measured (40/slice smoke): 8.8k-26k tokens, median ~17.5k (real MemoryCode long items: median 15.6k, mean
19.9k). The current session is capped at 1,500 estimated tokens (asserted <= 2,300 exact) so it fits the window
even with a 1,024-token reminder. prepare_report.json records the exact distribution per slice.

## Rendering (the Codex-authored frozen paraphrase set, `paraphrases.json`)
Authored once by Codex (gpt-6-sol, containerized, `build/gen.sh`, prompts in `build/prompts/`, raw outputs in
`build/raw/`), validated by `build/assemble.py` (one splitter span for every slot value and frame, no 5-word run from
topics.json, placeholders exact, ASCII), then frozen. Contents:
- `conventions`: 9 verb-phrase templates per kind (case, chx, prefix, suffix, annotation, try, assert, docstring,
  comment, import, decorator, digit) with slots `{obj} {objs} {affix} {module} {decorator} {case}`.
- `fillers`: 3 verb-phrase paraphrases per filler-instruction variant (79 variants).
- `codetalk`: 90 code-advice verb phrases that are not conventions; 150 standalone mentor sentences about code.
- `frames`: 45 single (`{vp}` at the start, mid-sentence, or after a lead-in), 24 double (`{vp1}`, `{vp2}` in one
  sentence), 30 "update" frames (announce a change without naming the old value).
Every list is split into disjoint thirds by a seeded permutation (validation / holdout / pilot); filler
paraphrase k belongs to slice k. The SAME frames wrap conventions, filler instructions and code advice, so no frame
marks a convention.

Per session (sentences = "units"): convention events -> if two events, one compound double-frame sentence with
P 0.4, else each alone; a convention or filler instruction shares its double frame with a code-advice VP with
P 0.15; update events use an update frame with P 0.5; plus 0-2 code-advice sentences and 1-3 code-talk sentences
(drawn without replacement per dialogue). Units are shuffled and each is inserted into a random mentor turn of the
session at a random sentence boundary (start of turn or after a `.!?`-terminated span). PREPARE asserts every unit
is exactly one `stencil.focus3.sentences` span of a mentor line of its session.

## Labels (private) and the oracle
`labels.jsonl` keeps the template, event kinds, filler events, units (text, session, kind, conventions carried),
live set, `history_regex`, `history_eval_query`, `families`, `required`, `structure`, exact lengths. `oracle_units`
= for every live (pivot, update), the unit that stated it (its latest statement), if it lies in sessions 0..s-1.
`build/oracle_plans.py` maps them to sentence ids for the pilot's label-derived oracle arm (<= 1,024 tokens).

## Ceiling test (tests/test_pack.py)
Three frozen trivial selectors over mentor sentences of sessions 0..s-1 — keyword (naming/annotation/try/assert/
docstring/comment/import/decorator/digit/case/always/never words), quoted token (`'x_'`), frame cue ("please",
"make sure", "from now on", ...) — scored against the oracle units: F1 must be < 0.8 on both splits, else the bank
is regenerated. 40/slice smoke: keyword ~0.2, quote ~0.4-0.7, cue ~0.2.

## Free real MemoryCode dialogues (holdout, descriptive only)
Recomputed from the snapshot with stencil's own item rules (`enumerate_items`/`long_items` criteria, Qwen3-4B
tokenizer, `split_items` seed 0): 108 short-eligible dialogues, 80 in stencil's derived pool, 212 long, 0 overlap,
28 free. PREPARE uses them only if these counts equal the reviewed accounting exactly (else it drops them and says
so in prepare_report.json). Item = the dialogue's first short-eligible session (as `split_items` picks), query =
its first `history_eval_query`; RUN at W = 1,536; reported as `real_fraction_required`, never primary or a guard.
