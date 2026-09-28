Found issues in all four categories. No files were modified.

**Evaluator gaming**

- [harness.py:45](/pack/frozen/run/harness.py:45), [evaluate.py:52](/pack/frozen/eval/evaluate.py:52): Surface code executes inside the measuring process and can patch timers, hooks, globals, or output writers; EVALUATE trusts the resulting statistics and NLL arrays. **Fix:** restrict surface changes to validated model configurations and generate measurements/artifacts in trusted execution.
- [loop.py:75](/pack/surface/loop.py:75), [harness.py:125](/pack/frozen/run/harness.py:125): Zero gates make extra passes irrelevant to predictions while satisfying `depth_ratio`; discarded layer calls also count. **Fix:** enforce the recurrent computation in frozen code and require a loop-disabled ablation demonstrating that recurrence contributes.
- [program.md:12](/pack/program.md:12), [IDEA.md:55](/pack/IDEA.md:55): Candidates can improve LR, rank, or schedule against the original baseline while adding ineffective loops, attributing ordinary tuning gains to recurrence. The matched ablation is only a recommendation. **Fix:** require every finalist to beat its own identically tuned, loop-disabled arm under the same budget.
- [harness.py:112](/pack/frozen/run/harness.py:112), [progen.py:68](/pack/frozen/run/progen.py:68): Arbitrary `logits()` code can interpret the prompt’s Python program and force the answer tokens, with dummy decoder calls satisfying depth. **Fix:** make frozen code compute logits from the approved neural architecture, excluding symbolic answer computation.

**Leakage**

- [prepare.py:29](/pack/frozen/run/prepare.py:29), [prepare.py:41](/pack/frozen/run/prepare.py:41), [prepare.py:63](/pack/frozen/run/prepare.py:63): Fixed seeds, generation order, deduplication logic, and the answer-producing generator all reside in RUN. Surface code can reconstruct validation and holdout labels without opening private files. **Fix:** generate splits using private randomness and keep preparation code/seeds outside RUN.
- [harness.py:47](/pack/frozen/run/harness.py:47), [prepare.py:116](/pack/frozen/run/prepare.py:116): Complete evaluation prompts and retention sequences are available before training; surface code can train on them or memorize retention targets through process globals or public files. **Fix:** isolate training from evaluation data and serve evaluation inputs only after locking an immutable checkpoint.

The program generator does deduplicate training, validation, and holdout programs across splits; that does not prevent the reconstruction and access paths above.

**Budget loopholes**

- [harness.py:45](/pack/frozen/run/harness.py:45), [harness.py:83](/pack/frozen/run/harness.py:83), [harness.py:101](/pack/frozen/run/harness.py:101): Import-time work precedes the training timer, and `logits()` runs after budget accounting ends; either can perform additional training. `no_grad()` can be locally overridden. **Fix:** externally meter all surface execution and prohibit parameter updates after the training deadline.
- [harness.py:105](/pack/frozen/run/harness.py:105): Trainable parameters are counted only after training; a candidate can update the full model, then clear `requires_grad` and report zero trainable parameters. **Fix:** enforce the permitted parameter set throughout training and verify frozen weights remain unchanged.
- [common.py:79](/pack/frozen/run/common.py:79), [harness.py:125](/pack/frozen/run/harness.py:125): Depth counts module invocations, regardless of internal batch/sequence expansion; direct `.forward()` calls bypass hooks entirely. Extra inference computation can therefore escape the depth cost bound. **Fix:** enforce execution dimensions and operations in frozen code and meter actual inference resources.

**Scorer bugs**

- [evaluate.py:69](/pack/frozen/eval/evaluate.py:69): NLL validation accepts arbitrary token-axis lengths and negative values; empty arrays pass finiteness checks and yield NaN means. **Fix:** require the exact expected shape, nonnegative values, and finite aggregate NLLs.
- [evaluate.py:80](/pack/frozen/eval/evaluate.py:80): Calling `float()` accepts NaN, infinity, and negative guard statistics; these can be emitted with `valid: true`. **Fix:** validate finiteness and appropriate ranges before publishing metrics.
- [evaluate.py:100](/pack/frozen/eval/evaluate.py:100): A zero baseline NLL passes validation and causes an uncaught division-by-zero exception. **Fix:** validate the denominator and return a structured invalid result when it is nonpositive.

Scorer cases were confirmed by executing the evaluator with in-memory dependency fixtures. GPU/model tests were unavailable; external runner enforcement is outside the supplied pack.
---
## Orchestrator response (2026-09-27, round 2)

**Evaluator gaming**
- *Surface patches timers/hooks/globals in-process*: **bounded**. The surface is now only a loop `nn.Module`. It never receives the base model, the layers, token ids, the optimizer or the data. The frozen runtime (`frozen/run/looprt.py`) owns the forward, the loss, the optimizer and the schedule. Depth is metered inside the frozen `block` callable, not by hooks. Every parameter and buffer is hashed after training and again after decoding and the NLL passes, and the base weights are hashed at load and at the end; any mismatch is `invalid`. What remains is deliberate in-process monkey-patching (e.g. of `time` or `torch`). That is forbidden by `program.md`, visible in every kept diff, and outside the PLAN §3.7 threat model (good-faith proposer).
- *Zero gates / discarded block calls satisfy depth*: **fixed**. The harness also decodes the main set with the loop disabled: the frozen code runs `block(p)` once, with the same trained weights. The new guard `loop_gain ≥ 0.005` requires the extra passes to improve accuracy; zero gates or dummy calls give exactly 0. Batch or sequence expansion is **fixed** too: `block(x)` only accepts x of exactly p's shape (tested).
- *Tuning gains attributed to recurrence*: **fixed**. LoRA rank, alpha and targets, LR, betas, WD, warmup/cosine schedule, grad clip, batch, data order and the answer loss are frozen in `looprt.py`/`harness.py` at the former baseline values. The surface may change only the loop window and counts, the update rule, injection, the loop's own parameters, init and LR, halting, and auxiliary losses that act through the loop (`ctx.readout`, `ctx.aux_loss`). Together with `loop_gain`, every candidate is compared with its own loop-disabled arm.
- *Symbolic answering in `logits()` + dummy depth*: **mostly fixed, rest bounded by program.md**. `logits()` no longer exists. The frozen code computes logits, and the loop sees only the layer-12 hidden state. It cannot read the program text except by decoding hidden states, which `program.md` forbids along with dummy `block()` calls. A symbolic shortcut would also have to survive `loop_gain` and the causality check.

**Leakage**
- *Generator, seeds and dedup reachable from RUN*: **fixed**. `progen.py`, `prepare.py` and the seeds moved to `frozen/prepare/`, which is mounted only into PREPARE (`python /prepare/prepare.py`) and the pack tests (`/pack`). It is still part of `data_hash`. RUN mounts `frozen/run/` = `common.py`, `harness.py`, `looprt.py` only; a test asserts that no RUN file contains the generator or the seeds.
- *Eval prompts and retention text visible before training*: **bounded / by design**. RUN needs the prompts to decode. The harness now loads `items.json` only after training, and the surface has no data paths or handles. Reading `/data/public` from `loop.py` is forbidden by `program.md`. The prompts carry no answers, and private data is never mounted in RUN. Retention text is loaded before training for the base-model NLL. Memorizing it is test-time training on validation data (forbidden) and would show as a `text_nll` drop that the diff explains.

**Budget loopholes**
- *Import-time work before the timer; learning after the deadline*: **fixed**. The timer starts before `import loop`. `unchanged_after_train` (hash of every parameter and buffer after training vs after all decoding and NLL passes) makes any post-deadline change `invalid`. `no_grad` overrides cannot matter, because the frozen optimizer is never stepped after training and any change is caught by the hash.
- *Trainable set counted only at the end*: **fixed**. `Trainer.check_trainable()` runs before every step and once after training. The optimizer's set must equal the LoRA params plus `loop.parameters()`, no other parameter may require grad, and the loop may not hold base or LoRA parameters (tested). `base_unchanged` (sha256 of all base weights at load vs end) makes any base edit `invalid` (tested).
- *Depth counts invocations; `.forward()` bypasses hooks*: **fixed**. The loop has no access to layer modules. Depth is metered as tokens through the frozen `block` callable, and `block` rejects inputs that are not exactly p's shape. Computation the loop does in its own parameters (e.g. a large MLP) is bounded by `trainable_m ≤ 18` and `train_s ≤ 1.3×`.

**Scorer bugs**
- *NLL arrays with any token length, negative values or empty input*: **fixed**. Both arrays must have exactly the (rows, TEXT_LEN) shape of the public text, be non-empty, be finite, and be ≥ 0 (tested).
- *`float()` accepts NaN/inf/negative guard stats*: **fixed**. `depth_ratio`, `trainable_m`, `gen_s`, `train_steps` and `examples_seen` must be finite, in range, and integers where applicable (tested). Token ids must be real ints.
- *Zero baseline NLL → ZeroDivisionError*: **fixed**. It is a structured `invalid` (tested).
