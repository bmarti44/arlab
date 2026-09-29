# plastic-agent — gpt-6-astra review (round 1)

Found issues in all four categories. No files were modified. Numeric-parser and budget probes ran entirely in memory.

**Evaluator gaming**

- [harness.py:84](/pack/frozen/run/harness.py:84), [engine.py:183](/pack/frozen/run/engine.py:183), [trainer.py:45](/pack/frozen/run/trainer.py:45): The surface receives live model/budget handles and mutable adapters; it can change counters, forge tensors or training metadata, and still pass the adapter identity check. **Fix:** isolate the surface from GPU/model access and expose serialized RPCs with trusted accounting and immutable adapter IDs.
- [lora.py:159](/pack/frozen/run/lora.py:159): The “bit-for-bit” base-weight check is a weighted sum; permuting values within a parameter preserves it while changing model behavior across worlds. **Fix:** hash parameter bytes in a trusted process.
- [harness.py:53](/pack/frozen/run/harness.py:53), [evaluate.py:262](/pack/frozen/eval/evaluate.py:262): Candidates can select `ARM="placebo"`, changing which score becomes primary. All worlds’ public transcripts are loaded together, so the surface can train adapters for the preceding world and exploit the rotation while passing the prefill guard. **Fix:** choose reference/candidate roles outside surface-controlled code and force candidates to use their own-world score.

**Leakage**

- [build/pilot.sh:53](/pack/build/pilot.sh:53), [common.py:13](/pack/frozen/run/common.py:13), [prepare.py:92](/pack/frozen/prepare/prepare.py:92): The supplied RUN launch mounts the entire Hugging Face cache, including the GSM8K test questions and answers used for the private battery; the raw private-text source is also available there. **Fix:** mount only the required model/tokenizer files and approved replay data into RUN.
- [prepare.py:85](/pack/frozen/prepare/prepare.py:85), [prepare.py:96](/pack/frozen/prepare/prepare.py:96), [prepare.py:124](/pack/frozen/prepare/prepare.py:124): Validation and holdout reuse identical guard-world tasks, GSM8K items, and text-NLL rows, allowing validation-driven tuning against the holdout forgetting battery. **Fix:** generate disjoint batteries for each split.

**Budget loopholes**

- [trainer.py:129](/pack/frozen/run/trainer.py:129), [trainer.py:198](/pack/frozen/run/trainer.py:198): Negative `replay_rows` subtracts tokens when `kl_base` is positive, while replay computation is skipped; a 512-token batch with `replay_rows=-1000` charges **−513,488 tokens**. **Fix:** require finite, nonnegative integer replay counts and validate every charge.
- [engine.py:145](/pack/frozen/run/engine.py:145), [engine.py:47](/pack/frozen/run/engine.py:47): Generated tokens are charged after a batch with checking disabled; the final batch can exceed the cap without invalidating the run. **Fix:** reserve generation capacity or enforce the remaining allowance during decoding.
- [engine.py:118](/pack/frozen/run/engine.py:118), [engine.py:146](/pack/frozen/run/engine.py:146), [trainer.py:129](/pack/frozen/run/trainer.py:129): Accounting excludes padded positions and continued decoding of completed batch rows, although the model processes them. **Fix:** charge actual processed tensor positions, including inactive decoding rows.
- [harness.py:51](/pack/frozen/run/harness.py:51), [harness.py:78](/pack/frozen/run/harness.py:78): Surface import executes before any adaptation budget starts; import-time model work or background workers escape per-world accounting, subject only to the outer RUN timeout. **Fix:** start supervision before import, meter all surface descendants, and permit model computation only through the trusted service.

**Scorer bugs**

- [fauxos.py:393](/pack/frozen/prepare/fauxos.py:393), [fauxos.py:407](/pack/frozen/prepare/fauxos.py:407): Integer extraction accepts malformed answers: `answer("42e999")` and `answer("not 42")` both score as integer **42**. **Fix:** validate typed tool outputs and require an exact answer grammar for `answer(...)`.
- [scoring.py:7](/pack/frozen/eval/scoring.py:7): GSM8K parsing accepts numeric prefixes: `Answer: 42e9` and `Answer: 42/7` both score as **42**; a later malformed answer does not supersede an earlier match. **Fix:** fully parse the final answer line and reject trailing expressions or malformed replacements.
- [harness.py:99](/pack/frozen/run/harness.py:99), [harness.py:107](/pack/frozen/run/harness.py:107): The divergence guard checks only the returned adapter’s training stats. A surface can encounter non-finite loss, return an earlier clean adapter, and avoid invalidation. **Fix:** maintain a trusted, sticky divergence flag across every training call.

---
## Orchestrator response (2026-09-28, round 1)

**Evaluator gaming**
- *Live model/budget handles; mutable adapters, counters and metadata*: **partly fixed, rest bounded**.
  - Train returns an `Adapter` whose `tensors`/`cfg`/`stats` are copies. The trainer keeps its own copy of each adapter's (cfg, tensors, stats), and the harness saves only that copy (`Trainer.saved`). Editing or forging a returned adapter changes nothing on disk (tested: the probe surface overwrites its adapter with 7.0 and rank 64).
  - `init=` also resolves through the private copy. The evaluator re-validates every adapter's shape, config and finiteness.
  - What remains is deliberate in-process tampering with `gen`/`train` internals, such as `gen._budget` counters or the trainer object. `adapt()` runs in the harness process with live handles. As in the other packs, this is outside the PLAN §3.7 good-faith threat model: `program.md` forbids it, and it is visible in every kept diff.
  - Out-of-process RPC isolation of a 1.7B model on a shared-memory GPU is not cheap. It is not built.
- *Weighted-sum fingerprint misses permutations*: **fixed**. `lora.fingerprint` is now a sha256 over the name, dtype, shape and raw bytes of every base parameter and buffer. The harness checks it at load, before and after every world, and at the end. The test covers a swap of two elements (which the weighted sum missed) and a one-ulp change.
- *Candidate selects `ARM="placebo"`; all transcripts loaded together; the rotation is exploitable*: **fixed**.
  - No surface constant is read any more; `ARM` is gone from the surface and the references.
  - Frozen code (`common.arm_of`) picks the arm from the sha256 of `/work/adapt.py` against `frozen/run/ref_{none,icl,placebo}/adapt.py`. The four files are byte-distinct (tested). The harness and the evaluator each compute it, and the evaluator invalidates a run whose `stats.json` arm differs.
  - Every arm, the placebo reference included, is scored on its own worlds, and the primary is always own-world `success`.
  - The placebo reference is implemented in the harness: world i's `adapt()` receives world i+1's transcript and tool names, and the evaluator checks the recorded donors. For every other arm, donor = own world.
  - Public data is now `public/order.json` plus one file per world. The harness loads only the world being adapted and passes only that world to `adapt()` (tested via the harness globals).
  - A candidate's cross-world score (adapter j on world j−1) is reported only, as `cross_world_success` / `specific_gain`.

**Leakage**
- *RUN mounts the whole HF cache (GSM8K test, OASST2)*: **fixed in pilot.sh; bounded in campaigns**.
  - arlab itself always mounts the full cache read-only at `/hf` in RUN (`arlab/campaign.py` `trial()`: `common = [..., (HF, "/hf", False)]`). pack.yaml cannot narrow that, so the second remedy applies.
  - The GSM8K battery items and the OASST2 rows are copied into `private/` (and `train/replay.npy`) at PREPARE.
  - The dataset paths now live only in `frozen/prepare/prepare.py`. No RUN-side file (`frozen/run/*.py`, the references) and not the evaluator references `datasets--`, parquet or jsonl files; `common.py` names only the model snapshot (tested).
  - `build/pilot.sh` now mounts only `hub/models--Qwen--Qwen3-1.7B` at `/hf/hub/models--Qwen--Qwen3-1.7B` for RUN and EVALUATE.
  - What remains in a campaign: `adapt.py` could open the raw GSM8K parquet under `/hf` by file I/O. `program.md` forbids file reads, and such a read would be visible in the diff.
- *Validation and holdout share the forgetting battery*: **fixed**.
  - Each split has its own guard world: `g0` for validation and `g1` for holdout, from their own seeds in the guard range.
  - GSM8K items: 400 distinct test items are drawn once and split 200/200.
  - Text rows: validation `[0:64]`, holdout `[64:128]`, replay `[128:]`.
  - A test asserts all three are disjoint across the splits and from replay.

**Budget loopholes**
- *Negative/non-integer `replay_rows`; unvalidated charges*: **fixed**.
  - `trainer.check_config` type- and range-checks every field. For example: `replay_rows` must be an int in [0, 64], `lr` a finite number in (0, 0.1], and epochs, weights and clip must be finite and in range. `batch_size`/`max_len` must be ints (not bools), and `grad_ckpt` a bool. The LoRA rank, alpha, targets and layers are checked too.
  - Example weights must be finite and in [0, 1000].
  - Every `Budget` charge or refund must be a non-negative int, else ValueError (tested).
- *Gen tokens charged after the batch, unchecked*: **fixed**. Each generation batch reserves `B × (padded prompt block + max_new)` before it runs. A reservation over the cap or past the deadline raises `BudgetExceeded` with nothing charged, and unused decode steps are refunded afterwards. The cap therefore can never be exceeded (tested: repeated calls stop at the cap and never pass it; a batch one position short is refused up front). Teacher batches are charged `B × padded length` before they run. The prefix is charged once per context.
- *Padding and finished rows not charged*: **fixed**. The gen charge is exactly `B × (Ls + decode steps run)`, which includes padding and rows that already stopped. The train charge per step is `batch × padded length`, plus `2 × replay_rows × (replay length − 1)` when `kl_base > 0` (tested: the train charge is a multiple of the batch size).
- *Import runs before the budget*: **fixed** for the budget, **bounded** for background workers.
  - The model loads first. Then world 0's clock starts, the surface is imported, and the import time is charged to world 0's deadline, `adapt_s` and `adapt_s_max` (tested with a 3 s import-time sleep).
  - Threads or processes that the surface starts and leaves running are not metered per world. They stay under the RUN timeout and `peak_mem_gb`, and `program.md` forbids them.

**Scorer bugs**
- *`answer("42e999")` / `answer("not 42")` scored as 42*: **fixed**.
  - `fauxos.run_program` records a typed value per line (`fauxos.value_of`).
  - `answer(x)` accepts an int literal, or a string that is exactly an integer or a comma-separated integer list. `42e999`, `not 42`, `42.0`, `~42`, `42/1` and `42 or 43` give no value.
  - Tool outputs are parsed with their op's exact output grammar.
  - `score` compares the last line's typed value: an int must be a real `int` (not bool/float), and a set must be the same integers without duplicates (tested).
- *GSM8K parser accepts numeric prefixes; an earlier match survives a later malformed line*: **fixed**. The last `Answer:` line wins, and its remainder must fully match one number: optional `$`, thousands commas, optional decimals, an optional trailing period. `42e9`, `42/7`, `not 42`, `42 apples` and `1,23` give None, and so does `Answer: 18\nAnswer: 18e0` (tested).
- *Divergence guard only on the returned adapter*: **fixed**.
  - The trainer keeps a sticky `nan_seen` flag. It is set by a non-finite loss, a non-finite gradient norm or a non-finite adapter in any `train()` call. The harness ORs it across worlds into `stats.nan` and per-world `nan_seen`.
  - The evaluator invalidates the run on either. The test covers a finite adapter returned after a diverging second call.
  - The flag lives in the trainer object, so deliberately resetting it in-process falls under the first bounded item.

Changing frozen code means new data: the stale `~/arlab-data/plastic-agent/e8b29e06821e4352` was deleted after `arlab check --static` built `da29c4038bdbef14`. 20 CPU tests pass through `arlab check` (~2.5 min).

---
# gpt-6-astra review (round 2) and orchestrator response (2026-09-29)

Round 2 findings (review of v1 after round 1): `train.__self__` exposes `_kept` / `nan_seen`; world-specific
improvement only reported; campaigns mount the whole HF cache; all current-split transcripts readable and one surface
module persists across worlds; templates shared across splits; surface gets the raw engine and a mutable budget
(`budget=None`, counters, deadline); threads / processes / exit callbacks outlive `adapt()`; `prefill_tokens` ignores
retry prompts. Direct private-task leakage, scorer exploits and ICL-reference unfairness: none found.

**Isolation and budget (findings 1, 4, 6, 7): fixed**, with the latent-arch / ttc-controller patterns.
- `harness.py` is now a trusted supervisor that owns the model and never imports the surface. Per world it starts a
  fresh `child.py` (own session, minimal env, CUDA hidden, single-threaded BLAS) that applies Landlock (the latent-arch
  `sandbox.py`, plus single-file rules for `/dev/urandom`, `/dev/random`, `/dev/null`, which importing torch needs):
  readable only the Python install, system dirs, `/frozen`, `/work`; writable only a fresh per-world scratch dir that
  the supervisor deletes afterwards; no GPU, TCP, outside signals or ptrace. Its self-check (every RUN data file, the
  model snapshot, `/proc/<parent>/mem`, `/tmp` write) must pass or RUN crashes.
- The child gets only its world's transcript and tool names over stdin; `gen` / `train` are RPC stubs
  (`surface_api.py`). The supervisor validates, runs and meters every request (`engine.Gen`, `trainer.Trainer`, one
  `Budget` per world on CLOCK_MONOTONIC from "go", so every world pays its import). The child cannot pass `budget=None`
  (extra fields are ignored), touch counters, or move the deadline (its `seconds_left` / `tokens_left` are local copies).
  Adapters and teacher log-probs never enter the child: they stay in the supervisor under immutable integer ids;
  `adapt()` returns an id, and the supervisor saves its own copy (data-only safetensors, re-validated by the evaluator).
- At deadline + 15 s the supervisor stops serving and SIGKILLs the child's process group (the last adapter trained is
  kept). After `adapt()` returns it inspects the child from outside (new OS threads, descendants, orphans re-parented
  to the subreaper supervisor), records each as a violation (evaluator: invalid), then kills the group and every
  descendant BEFORE saving the adapter. SIGKILL means no atexit callback ever runs.
- Tests: a probe surface (both worlds, separate PIDs, empty module state, empty scratch) cannot read the other worlds,
  order.json, replay rows, the model snapshot, or the raw GSM8K / OASST2 files under `/hf`. It cannot list `/data`,
  `/hf`, `/tmp`, write `/work`, `/tmp`, `/out`, read the parent's mem / environ, open TCP, start a thread / fork /
  subprocess. Forged teacher and adapter ids, `budget=None`, unknown calls, bad types and negative `replay_rows` are
  refused, and its edits to returned stats change nothing saved. More surfaces: one that ignores the deadline while
  spamming calls, and a pure spin loop, are both killed at deadline + grace. A forged `AdapterRef` or a dict return
  crashes RUN. Local counter / deadline edits cannot exceed a 500-token gen cap. A ctypes pthread is caught (run
  invalid), and a double-forked `setsid` daemon is caught and killed. numpy / torch imports work in the sandbox with no
  violation. The sticky divergence flag and id validation are unit-tested in the supervisor-side trainer.
- Side fix: the old harness charged the first base-weight fingerprint (~20 s on the GB10 in gate 2) to world 0's
  `adapt_s`, and hashed before AND after every world. The clock now starts at "go", and the hash runs at load and
  after each world (n+1 instead of 2n+1, ~80 s less per validation RUN).

**World-specific learning control (finding 2): fixed.**
- For candidates, the evaluator scores every target world i with the adapter trained on a DIFFERENT world (donor
  i+1) of the same split. The old cross-world pass, renamed, reuses the merge of adapter i+1. It reports
  `mismatched_success`, `world_specific_gain` = success − mismatched_success and `world_specific_share` =
  (success − mismatched) / (success − none_success), defined as 1.0 when success − none_success < 0.025.
- New guard `world_specific_share >= 0.5`. The guard is checked only on candidates, never on the baseline or the
  references.
- Tests: the formula's cases (`scoring.world_specific_share`), and the end-to-end metrics consistency.
- Noise: the paired SE of success − mismatched on 240 items is ~0.02–0.03, so the guard can reject a real but small
  world-specific gain by chance (IDEA.md caveat 5).
- Extra EVALUATE time: the control is one more greedy pass over the split's tasks with an already-merged adapter.
  Estimate from gate 2 (240 plain tasks at batch 48 × 64 decode steps, plus retries): ~10–30 s per validation run and
  ~25–80 s per holdout run. A whole candidate EVALUATE is est. ~3 min validation, ~5–6 min holdout. The first run per
  campaign adds ~2 min to fill the base cache.
- Decision: keep `evaluate.timeout_s: 3000`. It is 10× the estimate, and the pass already existed as the reported
  cross-world score, so the evaluator's cost did not change.

**Token accounting (finding 8): fixed.** Per task, the evaluator records the full context of each model call (the
shared prefix counted in each), for the first call and for the retry, plus the generated tokens. It reports:
- `initial_context_tokens`;
- `retry_prompt_tokens`;
- `prefill_tokens` = their sum (the guard now includes retries; the surface arms stay ≤ ~400, so max 1000 is unchanged);
- `decode_tokens`;
- `inference_tokens`.

The base-model cache stores these too (EVAL_VERSION pa-eval-3), so none-arm and no-adapter worlds count their retries.
Tested: the sums are consistent, and the retry context is counted when the tiny model retries.

**Recorded, not fixed (findings 3, 5).** IDEA.md caveats 7 and 8, with a test that the statements are there:
- The claim is limited to unseen worlds, not unseen task types: the 15 templates, their wording and the operation
  types are shared across splits, and a test asserts the template sets are equal.
- Campaigns mount the whole HF cache at `/hf` (runner level). The pack no longer relies on the GSM8K / OASST2
  sources being unreadable: the surface sandbox denies `/hf`, and the test opens exactly those two source files from
  inside the sandbox.

**Untested without the GPU:**
- The whole path on CUDA as uid 1000: Landlock + CUDA parent + per-world `Popen`; per-RPC `/proc` scans; JSON RPC
  overhead for large example lists.
- The real adapt / EVALUATE timings (the estimates above).
- The child's startup under the real image as non-root. The TESTS step runs as root; ttc-controller already runs a
  Landlocked child as uid 1000 in RUN.
- Whether the naive baseline's `world_specific_share` would pass. The guard does not apply to it, but it shows what
  the guard means at this scale.

Pack tests: 25 CPU tests pass.
