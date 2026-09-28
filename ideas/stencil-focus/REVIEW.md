Found one critical trust-boundary failure and several pilot/scoring weaknesses. No files were modified.

**Evaluator gaming**

- **Critical — surface code can fabricate completions.** [harness.py:40](/pack/frozen/run/harness.py:40) executes the surface inside the harness process; [evaluate.py:141](/pack/frozen/eval/evaluate.py:141) verifies the prompt but cannot authenticate the completion. A surface can replace `urllib.request.urlopen`, supply convention-compliant code, and falsify finish/token metadata. An in-memory probe confirmed the actual harness accepted this with zero recorded tokens. **Fix:** execute the planner in an isolated process that returns only serialized Plans; keep generation and output recording exclusively in the trusted parent.

- **The anti-memorization check only examines literal source words.** [evaluate.py:57](/pack/frozen/eval/evaluate.py:57) accepts encoded or dynamically assembled convention/task tables, despite the prohibition on memorized lists. **Fix:** treat the scan as lint, and use freshly randomized convention values/tasks to test generalization.

**Leakage and split design**

- **None found in the supplied construction code under the stated mounts.** [prepare.py:176](/pack/frozen/run/prepare.py:176) separates public text from private labels; [prepare.py:207](/pack/frozen/run/prepare.py:207) copies the vendor material only into private storage. There is no training split.
- Validation/holdout pivots are disjoint and cover all 51 pivots. I checked the actual frozen convention, filler, advice, talk, and frame pools: no exact cross-slice overlap. Chatter is partitioned before consumption and source overlap is asserted at [prepare.py:210](/pack/frozen/run/prepare.py:210). This establishes source-ID separation, not content deduplication across source datasets.
- The omitted public stencil modules could not be inspected for embedded private material.

**Budget loopholes**

- **Auxiliary computation is not confined to the budgeted service.** [harness.py:40](/pack/frozen/run/harness.py:40) permits arbitrary imports and computation; [pack.yaml:13](/pack/pack.yaml:13) budgets only service tokens. Local auxiliary inference/parallel CPU work would not appear in vLLM counters, although runtime and memory limits still apply. **Fix:** isolate the planner with explicit CPU limits, tokenizer-only assets, and no model, network, or subprocess access.
- **No service-token undercount found given your external counter accounting.** The harness’s incomplete accounting of retries/timeouts is therefore not an exploitable billing loophole.

**Scorer bugs**

- **First-code-block-only scoring leaves later code unchecked.** [test_pack.py:253](/pack/tests/test_pack.py:253) explicitly documents and preserves that behavior. A compliant first block can conceal a later implementation that violates conventions. **Fix:** score all emitted code blocks together, or reject multiple blocks. This behavior is documented by the supplied test; the omitted checker prevented direct reproduction.

**Baseline**

[surface/stencil.py:15](/pack/surface/stencil.py:15) matches the stated Exp 4C policy: evicted mentor sentences, newest-first packing into 256 tokens, chronological rendering, header 0, after-thread placement. [window.py:149](/pack/frozen/run/window.py:149) correctly stops at the first non-fitting sentence. Byte-for-byte upstream equivalence remains unverified because the snapshot is absent; the supplied equivalence test is at [test_pack.py:199](/pack/tests/test_pack.py:199).

**Pilot oracle and go/no-go**

- **GO is possible when every cap fails.** [pilot_report.py:53](/pack/build/pilot_report.py:53) falls back to `CAP` when no cap satisfies the 5% limit; the GO condition ignores this failure. An in-memory run with 100% cap hits in every arm returned `GO: 300/300`. **Fix:** require a nonempty acceptable-cap set before issuing GO.

- **A smaller recommended cap is never revalidated.** [pilot_report.py:52](/pack/build/pilot_report.py:52) recommends a lower cap while retaining headroom and failure measurements from the larger-cap run. Truncation can change both. **Fix:** rerun all arms at the selected cap before sealing the decision.

- **The oracle can restate superseded instructions.** [bank.py:380](/pack/frozen/prepare/bank.py:380) selects a compound sentence whenever one contained convention remains live; another convention in that sentence may be obsolete. [oracle_plans.py:25](/pack/build/oracle_plans.py:25) also packs optional and required conventions indiscriminately. This can depress the oracle and cause a misleading NO-GO. **Fix:** make oracle selection query-aware and conflict-aware, and report clipping/stale-rule coverage rather than treating it as a ceiling.

- **Pilot command failures can be masked.** [pilot.sh:8](/pack/build/pilot.sh:8) omits `pipefail`; [pilot.sh:23](/pack/build/pilot.sh:23) and [pilot.sh:52](/pack/build/pilot.sh:52) run `chown` after Python regardless of failure. Existing oracle plans can survive a failed regeneration. **Fix:** enable `pipefail`, use `python … && chown …`, and clear stale oracle artifacts before generation.
---
## Orchestrator response (2026-09-27)
Review: one read-only `gpt-6-astra` call (docs/ORCHESTRATOR.md step 3) on a copy of the pack without the snapshot
tarball and without data.
- **Surface can fabricate completions (critical): fixed.** The surface now runs only in a separate planner process
  (`frozen/run/planner.py`) that returns serialized plans; the harness never imports it, re-checks each plan with its
  own Window, builds every prompt, calls vLLM and writes the outputs. Verified with a surface that monkeypatches
  `urllib.request.urlopen` (no effect on the recorded outputs); `tests/test_pack.py::test_surface_runs_only_in_the_planner_process`.
- **5-gram scan is only a lint: bounded by design.** IDEA.md specifies this check as the rule; what makes a
  memorized convention list useless is the bank: conventions are paraphrased (never verbatim topics text, tested),
  and the holdout uses pivots disjoint from validation, so a table fitted on validation pivots cannot transfer.
- **Auxiliary computation / direct model calls from the planner: bounded.** program.md forbids model loads and LLM
  calls; CPU work is bounded by the `train_s` 1.3x guard and `run.timeout_s`; any direct vLLM call is counted by the
  runner's service-token measurement against `budget.limit`. Residual slack (the limit assumes every generation
  hits the cap) is accepted under the good-faith proposer model (PLAN §3.7), as in memory-longmemeval.
- **First code block only: by design.** That is the vendored, unmodified MemoryCode checker (`extract_code`) and
  stencil's `fraction_required`; changing it would change the pre-registered metric. The surface can only choose
  verbatim earlier sentences, so it cannot instruct the model about output formatting.
- **Pilot GO possible with no acceptable cap: fixed** (`pilot_report.py` now returns NO-GO). **Smaller cap not
  revalidated: fixed** as a report note (re-run `pilot.sh <dir> <cap>` at the recommended cap before sealing).
- **Oracle restates superseded/optional conventions: fixed/reported.** The oracle keeps only live statements whose
  family the query requires; compound sentences that also carry a superseded convention are kept (the live
  statement exists only there) and counted in the oracle script's output.
- **pilot.sh masks failures: fixed** (`set -euo pipefail`, `python … && chown`, stale oracle plans deleted and
  their presence checked before the arms run).
