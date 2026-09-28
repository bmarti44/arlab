# ttt-context — independent review (gpt-6-astra, round 1) and the orchestrator's answers

## Review (verbatim, `/tmp/arlab-review-ttt-context/review.md`)

Found issues in all four categories. `/pack` was not modified.

- **Evaluator gaming — decoding hooks can manufacture answers.** [tttlib.py:205](/pack/frozen/run/tttlib.py:205) runs greedy decoding before hooks are cleared at line 147. A surface hook can overwrite logits with chosen tokens; reset and SHA checks still pass. **Fix:** remove all surface hooks before decoding in a trusted model instance.

- **Evaluator gaming — gains need not involve weight updates.** [tttlib.py:52](/pack/frozen/run/tttlib.py:52) exposes the same question tensor later used for answering; [tttlib.py:195](/pack/frozen/run/tttlib.py:195) accepts arbitrary caches. Prompt rewriting or cache-only retrieval can beat `no_ttt` with unchanged weights. **Fix:** preserve immutable answering inputs, restrict cache provenance, and compare against an otherwise identical arm with updates disabled.

- **Evaluator gaming — frozen files do not protect harness execution.** [tttlib.py:190](/pack/frozen/run/tttlib.py:190) executes surface code in the harness process, allowing replacement of `greedy`, timer/reset functions, or output-writing behavior; [evaluate.py:55](/pack/frozen/eval/evaluate.py:55) trusts reported hashes and reset flags. **Fix:** isolate adaptation from the trusted controller, decoder, measurement, and artifact writer.

- **Leakage — RUN can reconstruct hidden labels.** [prepare.py:23](/pack/frozen/run/prepare.py:23) exposes split seeds; [prepare.py:47](/pack/frozen/run/prepare.py:47) exposes item mapping; [docgen.py:225](/pack/frozen/run/docgen.py:225) deterministically constructs answers before distractors. I reproduced all 800 validation/holdout question–answer pairs using just eight distractors per item. **Fix:** keep generation code and secret generation seeds exclusively in PREPARE, outside RUN.

- **Leakage — state survives between items.** [tttlib.py:37](/pack/frozen/run/tttlib.py:37) removes only `ttt` from `sys.modules`; [tttlib.py:144](/pack/frozen/run/tttlib.py:144) leaves arbitrary model attributes and other module globals intact. Stored documents, optimizer state, or learned deltas can influence later items while tensor restoration passes. **Fix:** use a fresh isolated worker per item with immutable base weights.

- **Budget loophole — deferred surface computation is free.** Timing stops at [tttlib.py:192](/pack/frozen/run/tttlib.py:192), before custom cache methods and decoding hooks execute at lines 203–205. These callbacks can perform additional training or model generation charged only to `answer_s`. **Fix:** prohibit surface callbacks during answering and meter all surface-controlled execution.

- **Budget loophole — import work escapes measurement.** [tttlib.py:177](/pack/frozen/run/tttlib.py:177) measures import without a following device sync; asynchronous GPU work is absorbed into prefill. Surface-controlled flag conversion at line 182 is also outside both timed intervals. **Fix:** synchronize around import, validate flags as literal booleans, and charge flag evaluation to adaptation.

- **Scorer bug — normalization changes numeric meaning.** [scoring.py:18](/pack/frozen/eval/scoring.py:18) discards signs and percentage symbols: verified that both `-3` and `3%` score as correct for gold `3`. **Fix:** use answer-type-aware normalization that preserves numeric signs and units.

- **Scorer bug — malformed output validation is incomplete.** [evaluate.py:61](/pack/frozen/eval/evaluate.py:61) accepts booleans and token IDs outside the tokenizer vocabulary; decoding at line 80 is outside exception handling. Line 65 also accepts fractional `nonfinite` values such as `0.5` by truncation. **Fix:** require exact integer types, validate actual vocabulary membership, and convert decoding failures into invalid results.

Model-level execution was unavailable because this environment lacks PyTorch and Transformers; the generator and normalization reproductions used Python’s standard library.

---
## Orchestrator response (2026-09-28)

**Evaluator gaming**
- *Decoding hooks can manufacture answers*: **fixed**. The surface trains a separate WORK copy of the model; the ANSWER
  model is never passed to surface code. Before answering, the harness clears every global module hook and restores the
  class dictionaries of all model module classes, so no hook or patched method written during `adapt()` runs while
  answering. ANSWER is checked to still equal the pristine weights (a surface that reaches it through `gc` makes the
  item `reset_ok: false`, so the run is invalid). Tests: `test_surface_code_cannot_touch_answering` (module and global
  hooks forcing a token, patched `nn.Linear.forward`, patched `tttlib.greedy`, a replaced SDPA registry entry:
  answers equal no_ttt's) and `test_tampering_is_refused_or_flagged[answer_model]`.
- *Gains need not involve weight updates (question tensor reused, arbitrary caches)*: **fixed**. `adapt()` may return
  only `{"weights": {name: tensor}, "doc_in_context": bool}`; any other key, a cache, a non-bool flag, unknown names,
  wrong shapes/dtypes or non-finite values raise. The answer prompt is built by frozen code from token copies taken
  before `adapt()` runs, with the harness's own document cache (base weights; the surface gets a separate clone). So a
  gain can only come from weights, and `no_ttt` is exactly the arm with updates disabled (same prompt, same cache,
  empty `weights`). Tests: `test_bad_return_values_are_refused`, `test_returned_weights_are_what_answers`.
- *Frozen files do not protect harness execution*: **bounded**. The answer path uses functions bound before any surface
  code runs, restored module dictionaries (`tttlib`, `torch.nn.functional`, `modeling_qwen3`, `sdpa_attention`,
  `masking_utils`, `cache_utils`, `modeling_utils`) and the attention/mask registries; the output writer (`json.dump`)
  is bound before the item loop. Deliberate patching of objects that are not snapshotted (e.g. `torch.matmul`, `time`)
  remains possible in-process. It is forbidden by `program.md`, visible in every kept diff, and outside the PLAN §3.7
  threat model (good-faith proposer). The evaluator trusting the reported hashes and restore flags belongs to the same
  bound. A subprocess per item would reload the 1.7B weights per item, and CUDA IPC to a persistent trusted process is
  untested on the GB10, so we did not add one.

**Leakage**
- *RUN can reconstruct hidden labels (generator, seeds, item mapping)*: **fixed**. `docgen.py`, `prepare.py`, the
  split seeds, the item mapping, `CONTEXT_TOKENS`, `N_ITEMS` and `DIFFICULTY` moved to `frozen/prepare/`, which is
  mounted only into PREPARE (`python /prepare/prepare.py`) and the pack tests (`/pack`). It is still part of `data_hash`.
  RUN mounts `frozen/run/` = `common.py` (prompt format), `harness.py`, `tttlib.py` and the two reference arms. The seed
  values were removed from `IDEA.md` (agent-visible). Test: `test_run_cannot_regenerate_the_data` (no generator file,
  seed constant or generator code in any RUN file, `program.md` or `IDEA.md`).
- *State survives between items*: **fixed** for everything reachable by honest code, **bounded** otherwise. Per item:
  every RNG (`random`, `numpy`, `torch`, CUDA) is reseeded from the item seed; `ttt` is imported afresh; `sys.modules`
  entries the surface added outside site-/dist-packages and the standard library are deleted; WORK is restored
  (weights, buffers, hooks, instance attributes added or changed on any module, config, train/eval mode) and verified
  equal to the pristine weights; a thread left running raises. Stashes inside third-party modules (e.g. an attribute on
  `torch`) remain possible in-process (same bound as above). Test: `test_no_state_survives_between_items`.

**Budget loopholes**
- *Deferred surface computation (cache methods, decoding hooks)*: **fixed**. No surface object reaches answering: the
  return value is validated and every tensor cloned inside the timed region, and hooks/patches are removed before
  answering (see above).
- *Import work escapes measurement; flag evaluation untimed*: **fixed**. The device is synchronized right before the
  timer starts (after the harness's prefill) and after the return value is validated; the import, `adapt()`, the
  literal-bool check of `doc_in_context` and the weight cloning are all inside it. Test:
  `test_budget_includes_import_and_return_validation`.

**Scorer bugs**
- *Normalization changes numeric meaning*: **fixed**. Tokens are signed numbers with their `%` sign or alphanumeric
  words, so `-3`, `3%` and `3.5` no longer match `3`; punctuation between words still separates (`Dock Velm.` matches
  `dock Velm`). Test: `test_normalize_and_score`.
- *Malformed output validation incomplete*: **fixed**. Token ids must be exact `int`s (not `bool`/`float`) in
  `[0, len(tokenizer))`; `nonfinite` must be an exact `int` equal to 0; timings must be exact `int`/`float`; the
  tokenizer is loaded and every answer decoded inside the validation block, so a decode failure makes the run invalid.
  The end-of-run SHA of the WORK copy must also match. Test: `test_evaluator_enforces_budget_and_reset`.
