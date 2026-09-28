# stencil-focus

Does the *selection* of re-presented instructions matter for a frozen model? A label-free reminder policy
(`surface/stencil.py`) chooses which earlier verbatim sentences of a long mentor/mentee conversation are restated
inside a 3,584-token window for Qwen3-4B; the metric is stencil-llm's `fraction_required` on a fresh synthetic
MemoryCode-style bank. Pre-registration: `IDEA.md` (review: `IDEA-REVIEW.md`). Bank construction:
`frozen/prepare/BANK.md` (not agent-visible).

**Licensing: local use only.** The pack wraps GPL-2.0 code from `~/stencil-llm` (5 modules of `src/stencil/`) and
Apache-2.0 MemoryCode material, taken from a pinned, path-limited snapshot (`frozen/prepare/stencil-<sha>.tar.gz`,
git-ignored, regenerate with `build/snapshot.sh`). `~/stencil-llm` itself is never modified.

## Layout
| path | role |
|---|---|
| `surface/stencil.py` | the editable policy; baseline = stencil Exp 4C (evicted mentor sentences, newest-first, 256 tokens) |
| `frozen/run/window.py` | agent-visible: sentence records, `Plan`, token counting, packing, displacement, exact prompt |
| `frozen/run/planner.py` | RUN subprocess: the only place the surface runs; returns serialized plans |
| `frozen/run/harness.py` | RUN (trusted, never imports the surface): re-checked plan -> prompt -> one greedy vLLM completion per item |
| `frozen/run/prepare.py` | PREPARE: snapshot, downloads, bank, splits (300/300 + 60-dialogue pilot slice + 28 real, holdout) |
| `frozen/run/ref_off`, `ref_evicted_1024` | references: no reminder; the baseline with a 1,024-token budget |
| `frozen/eval/evaluate.py` | EVALUATE: rebuild + verbatim/window/cap checks, topics.json 5-gram check, vendored checker |
| `frozen/prepare/` | `bank.py`, `paraphrases.json` (Codex-authored once, frozen), `BANK.md`, the snapshot tarball |
| `build/` | `snapshot.sh`, `gen.sh` + `prompts/` + `raw/` + `assemble.py` (paraphrase set), `pilot.sh`, `oracle_plans.py`, `pilot_report.py` |
| `tests/test_pack.py` | ceiling (F1 < 0.8), g0 exclusion, 4C byte-for-byte, stratification, splitter, scorer, anti-cheat |

## Before the first campaign
1. `build/snapshot.sh` (only if the tarball is missing; PREPARE checks its sha256).
2. `~/arlab/.venv/bin/arlab check --static ideas/stencil-focus` (PREPARE downloads Qwen3-4B ~8 GB into the HF cache).
3. GPU pilot: `build/pilot.sh /tmp/stencil-pilot` -> go/no-go + generation cap + `run.timeout_s`; record in DECISIONS.md,
   update `--max-new` (run and evaluate), `run.timeout_s` and `budget.limit` in pack.yaml before sealing.
4. Astra review (docs/ORCHESTRATOR.md step 3; see REVIEW.md), full `arlab check`, then `arlab run`.
