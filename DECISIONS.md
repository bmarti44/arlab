# DECISIONS (one line per decision, with the reason)

- 2026-09-26 Pasted kickoff matched PLAN.md Appendix C / §2.2 step 5 → treated as the owner's hand-off and executed PLAN.md §0.
- 2026-09-26 Build arlab from scratch rather than fork helix: the runner-owned loop, sealing and stats are the bulk of the work and helix's agent-runs-loop model would need rewriting anyway (simplest path; prior-art §2.7 allows either).
- 2026-09-26 Agent image adds python3/git/ripgrep on top of node:22-bookworm + codex 0.157.1: Codex tools expect a shell with rg/git; py3 lets the agent py_compile its own edits.
- 2026-09-26 M0 vLLM smoke uses Qwen3.5-4B, not Qwen3-4B: the HF cache holds only a refs stub for Qwen3-4B and Qwen3-30B-A3B (no weights); downloads are allowed only during PREPARE.
- 2026-09-26 peak_mem_gb = max over 5 s samples of (container memcg memory.current + GPU used_memory of the container's PIDs): measured that GB10 cudaMalloc is not charged to the memcg. mem_gb is used only for the wait-for-free gate, never as a docker --memory limit (a limit would not cover GPU allocations and could cause spurious OOMs).
- 2026-09-26 SEAL is done first into sealed.tmp; CHECK/PREPARE/TESTS then run from that snapshot and it is renamed to sealed/ only after TESTS pass: same phase semantics as the plan, but the prepared data provably comes from the sealed bytes.
- 2026-09-26 arlab/lib is sealed as sealed/arlab_lib/arlab/lib (all lib files, they are small) and mounted at /arlab_lib, so packs `from arlab.lib import x` with PYTHONPATH=/frozen:/arlab_lib.
- 2026-09-26 TESTS and PREPARE run without the GPU so `arlab check --static` never needs it; TESTS run as root (like EVALUATE) with the whole data dir mounted read-only at /data.
- 2026-09-26 Item packs: PREPARE writes /data/splits.json {"validation": n, "holdout": n} so the power check knows n_holdout_items before FINALIZE (no pack.yaml field added). Missing → config error.
- 2026-09-26 budget.json key = budget.unit; for unit `service_tokens` the runner's /metrics measurement is used. Missing or over the limit → invalid.
- 2026-09-26 EVALUATE exiting non-zero → `invalid` (the evaluator is frozen; failures to load/score the surface's outputs are the candidate's fault); RUN non-zero → `crash`; py_compile failure of an edit → `crash` (fix path applies).
- 2026-09-26 Calibration and holdout trials live in calib/ and holdout/ (result.json per seed); runs/ holds only LOOP experiments. Baseline/reference failing in CALIBRATE, or the pack failing CHECK/PREPARE/TESTS, is a planned stop with `config error` in the report (exit 0).
- 2026-09-26 Test hooks kept minimal: ARLAB_DEBUG_KILL=<during_run:ID|after_record:ID> (runner SIGKILLs itself ≡ kill -9), ARLAB_BACKOFF_S (infra backoff base), ARLAB_RUNS/ARLAB_DATA roots, hidden `--script` for the ScriptedBackend.
- 2026-09-26 Fixture pack: numpy MLP, 1500 steps × batch 1024 (~1.1 s train, ~2 s container), baseline lr 0.002 (acc 0.348 ± 0.006), known better lr 0.1 (+0.21). Default is a seed pack; tests make items / deterministic / MES variants.
