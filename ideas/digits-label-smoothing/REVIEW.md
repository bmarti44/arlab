- **Evaluator gaming / leakage — [prepare.py:12](/pack/frozen/run/prepare.py:12), [prepare.py:14](/pack/frozen/run/prepare.py:14):** Both evaluation splits come from the fully labeled, bundled `load_digits()` dataset using a reproducible permutation. Surface code can reconstruct validation and holdout labels and return perfect one-hot predictions without learning.
  **Fix:** Use genuinely private evaluation examples whose labels cannot be reconstructed from RUN’s dependencies or public data.

- **Budget loophole — [harness.py:21](/pack/frozen/run/harness.py:21), [harness.py:25](/pack/frozen/run/harness.py:25), [harness.py:29](/pack/frozen/run/harness.py:29):** The harness counts callback invocations and unconditionally reports 1,500 steps. Editable callbacks can perform multiple updates per call or train inside `build()` and `predict()`, exceeding the optimization budget while passing that accounting.
  **Fix:** Move optimizer updates and budget accounting into trusted execution code; restrict callbacks to model/loss computation.

- **Compute loophole — [harness.py:15](/pack/frozen/run/harness.py:15), [pack.yaml:6](/pack/pack.yaml:6):** Surface code can undo `torch.set_num_threads(1)` or start workers. Wall-time limits do not enforce the stated single-thread compute allowance.
  **Fix:** Enforce CPU quotas and aggregate process-tree CPU accounting outside the surface process.

- **Scorer bug — [evaluate.py:23](/pack/frozen/eval/evaluate.py:23), [evaluate.py:27](/pack/frozen/eval/evaluate.py:27):** Only loading is exception-protected. An NPZ archive named `logits.npy` lacks `.shape`; a correctly shaped string array makes `np.isfinite()` raise. Both escape without producing an invalid-result record.
  **Fix:** Validate ndarray type and numeric dtype, and include all artifact validation in the exception handler.

No files changed. Review was static with a dependency-free scorer control-flow check; NumPy, sklearn, and Torch were unavailable locally.
---
## Orchestrator response (2026-09-27)
- Label reconstruction via sklearn's bundled digits: harness now blocks `sklearn.datasets` before importing the surface; program.md forbids loading data. Residual (reading sklearn's data file directly, in-process) recorded in DECISIONS.md; every kept diff is visible in the report.
- Budget loophole (extra updates inside callbacks): by design the surface owns model/optimizer/step (as in nanochat-lite, PLAN §6.1); the frozen `train_s ≤ 1.5× baseline` guard bounds extra compute. Recorded, not changed.
- Thread count: the container is pinned to the fast cores; more threads would speed up, not exceed, the wall-time guard. Recorded, not changed (a CPU quota would be a runner change; arlab/ is frozen during M4).
- Scorer: all artifact validation is now inside the exception handler (non-numeric / npz → invalid).
