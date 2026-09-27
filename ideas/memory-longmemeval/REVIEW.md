Found issues in all four categories. No files were modified.

- **Evaluator gaming: gold-file symlink.** [evaluate.py:23](/pack/frozen/eval/evaluate.py:23) follows `answers.jsonl` symlinks. Surface code can register an exit handler replacing that output with a link to `/data/private/gold.jsonl`; it resolves inside EVALUATE, where gold rows already contain compatible `id`/`answer` fields ([prepare.py:59](/pack/frozen/run/prepare.py:59)). This supplies gold answers without solving questions. **Fix:** Reject symlinks and consume only securely opened, regular artifact files.

- **Leakage: answer-bearing turn annotations.** [prepare.py:55](/pack/frozen/run/prepare.py:55) copies dataset turns verbatim, including `has_answer`. I checked the pinned dataset: retained question `e47becba` contains two turns marked `has_answer: true`. Surface code can retrieve these directly. **Fix:** Reconstruct every public turn using an explicit allowlist containing only `role` and `content`.

- **Leakage: abstention labels encoded in IDs.** [prepare.py:27](/pack/frozen/run/prepare.py:27) derives the label from `_abs`, then publishes the original ID at [prepare.py:57](/pack/frozen/run/prepare.py:57). Although `answer()` receives no ID, it can inspect its caller’s `q` in [harness.py:36](/pack/frozen/run/harness.py:36). **Fix:** Publish opaque IDs unrelated to labels and isolate surface execution from harness memory.

- **Budget loophole: mutable accounting.** Surface code executes inside the trusted harness process ([harness.py:25](/pack/frozen/run/harness.py:25)), allowing replacement of `client.used`, `client.total`, and enforcement logic used at [harness.py:40](/pack/frozen/run/harness.py:40). An in-memory probe changed a simulated 200,001-token usage into an accepted answer reporting zero tokens. **Fix:** Keep enforcement, accounting, and artifact generation in a separate trusted process.

- **Budget loophole: direct model requests.** [harness.py:28](/pack/frozen/run/harness.py:28) connects directly to `http://llm:8000`; unrestricted surface Python can contact that same endpoint without `BudgetedClient`, bypassing its token and completion limits. The budget artifact records only client accounting ([harness.py:50](/pack/frozen/run/harness.py:50)). **Fix:** Expose only a trusted metering proxy to RUN and block direct service access.

- **Scorer bug: budget violations still score.** [evaluate.py:23](/pack/frozen/eval/evaluate.py:23) discards `status` and `tokens`, never checks `budget.json`, and declares readable submissions valid at [evaluate.py:34](/pack/frozen/eval/evaluate.py:34). A probe with `status: "budget_exceeded"` and 200,001 tokens received full credit. **Fix:** Validate trusted per-question and aggregate usage before scoring, rejecting budget violations.

- **Scorer bug: numeric normalization changes meaning.** [normalize.py:18](/pack/frozen/run/normalize.py:18) removes decimal points and minus signs; [normalize.py:22](/pack/frozen/run/normalize.py:22) then combines numeric tokens. Verified: `20.5` matches gold `25`, and `-5` matches `5`; conversely, `one hundred` fails against `100`. **Fix:** Preserve numeric punctuation and parse complete number expressions before general text normalization.

The external `BudgetedClient` and production orchestrator were not supplied; any additional protections there could not be verified.
---
## Orchestrator response (2026-09-27)
- Gold-file symlink: fixed in the runner for every pack — a RUN output containing a symlink that leaves the output
  dir is `invalid` before EVALUATE runs.
- `has_answer` leak: fixed — public turns keep only role + content. Also fixed a leak the review missed: original
  session ids mark evidence sessions (`answer_…`); public sessions are now `s0, s1, …`.
- `_abs` in ids: fixed — public/gold ids are opaque hashes (the original id is kept only in private gold).
- Scorer: `20.5`/`25` and `-5`/`5` collisions fixed (decimal points and minus signs kept); number words combine
  ("one hundred" → 100, "twenty one" → 21) but digit sequences no longer merge ("20 5" stays "20 5"); tests added.
- Budget violations: the evaluator scores only rows with status "ok" (over budget / crashed → 0).
- In-process accounting and direct model requests: by design the surface runs inside the frozen harness; the
  budget that decides validity (absolute service_tokens ≤ 160,000 × n) is measured by the runner from vLLM's own
  counters and cannot be bypassed from RUN. Recorded in DECISIONS.md.
