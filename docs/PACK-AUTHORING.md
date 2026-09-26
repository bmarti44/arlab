# PACK-AUTHORING — how to turn an idea into an arlab pack

A pack is a folder `ideas/<name>/`. The runner never changes for a new idea; everything idea-specific lives here.
Start with `arlab new <name> --idea <prose.md>` (copies `templates/pack/`), then fill in the files below.
Iterate with `arlab check --static ideas/<name>` (no GPU), finish with the full `arlab check ideas/<name>`.
Working examples: `ideas/_fixture` (CPU, seconds), `ideas/nanochat-lite` (GPU training),
`ideas/memory-longmemeval` and `ideas/agentic-coding-small` (eval-only with a vLLM service).

## 1. Files and contracts

| File | Role |
|---|---|
| `IDEA.md` | prose pre-registration: hypothesis, why, MES and its rationale, budget. Not machine-read. |
| `pack.yaml` | the only machine-read config (schema: PLAN §3.2; unknown fields are rejected). |
| `program.md` | the agent's instructions (≤ 80 lines, template in `templates/pack/program.md`). |
| `surface/` | the editable files — the only thing the agent changes. Keep it to 1–2 files. |
| `frozen/run/` | `prepare.py`, `harness.py` (RUN entry point), loaders, clients. Part of `data_hash`. |
| `frozen/eval/` | `evaluate.py` and scorers — mounted only into EVALUATE; the agent never sees it. |
| `frozen/prepare/` | optional: local source files (e.g. tasks with hidden tests) mounted only into PREPARE at `/prepare`. Part of `data_hash`. |
| `requirements.txt` | extra pip deps, installed on top of the base image with its `pip list` as constraints. |
| `tests/test_*.py` | deterministic CPU checks, run by `arlab check` as root with the data at `/data`. |

**RUN contract.** `run.command` always calls a frozen entry point with `{seed}` and `{split}`
(`validation` or, only in FINALIZE, `holdout`). The harness imports the surface from `/work`, drives it, and
writes everything the evaluator needs to `/out` plus `/out/budget.json` = `{"<budget.unit>": N}`.
The surface never reads argv, environment variables or data paths; the harness hands it what it needs.

**EVALUATE contract.** `evaluate.py --run /run_out --out /result/metrics.json` writes atomically
`{"valid": bool, "primary": float|null, "metrics": {...}, "items": {id: score}|null, "message": str}`.
- It computes every score itself from raw outputs (logits, answers, workspaces) and its own private data.
  Never trust a number the surface computed.
- `valid: false` → the run is `invalid` (use it for NaN/inf, wrong shapes, missing outputs, non-causal models).
- Means over questions/tasks **must** fill `items` (0/1 or float per item): that makes SE paired and honest.
- The runner adds `train_s`, `eval_s`, `peak_mem_gb`, `gpu_temp_max`, `service_tokens`, `budget_used`
  to `metrics` before guards are checked, so guards may name them.

**Mounts** (RO unless noted; `PYTHONPATH=/frozen:/arlab_lib`, so `from arlab.lib import …` works):

| Step | uid | Network | Mounts |
|---|---|---|---|
| PREPARE | 1000 | yes | `/frozen`, `/prepare` (if `frozen/prepare/` exists), `/data` **RW** (new dir), `/hf` **RW** (HF cache, `HF_HOME=/hf`) |
| TESTS | 0 | none | `/pack`, `/frozen`, `/eval`, `/arlab_lib`, `/work`, `/data` (whole data dir), `/hf` — no GPU |
| RUN | 1000 | none / campaign net | `/work`, `/frozen`, `/arlab_lib`, `/data/train`, `/data/public` (this split), `/out` **RW**, `/hf`, `/cache` **RW** |
| EVALUATE | 0 | same | `/frozen`, `/eval`, `/arlab_lib`, `/work`, `/run_out`, `/data/public`, `/data/private`, `/result` **RW**, `/hf`, `/cache` **RW** (separate) |

RUN never sees `private/`, `/eval` or the holdout (outside FINALIZE). `/run_out` is not `/run` because the
NVIDIA container hook needs a writable `/run`.

## 2. Data

PREPARE writes into `/data` once; the result is immutable and shared by every campaign with the same `data_hash`
(= prepare command + `frozen/run/` + `requirements.txt` + image digest). Layout:
```
train/                 training data (training packs)
validation/public/     inputs RUN may see (questions, task repos); may be empty
validation/private/    answers, hidden tests, eval tokens — EVALUATE only
holdout/public/ holdout/private/   the same for the one-shot holdout
splits.json            item packs only: {"validation": n_items, "holdout": n_items}
```
Rules: splits are disjoint and fixed (seeded split, pinned dataset revision / repo SHA inside `prepare.py`);
the holdout is never used for anything but FINALIZE; nothing in `train/` or `public/` may reveal a private answer.
Downloads go to the container's `/tmp` or `/hf`; keep only what the pack needs in `/data` (it is re-hashed on resume).
PREPARE and TESTS run without the GPU.

## 3. Statistics you must plan for (PLAN §3.5)

- **MES** (`metric.mes`) is the smallest effect you would call real. Fix it in `IDEA.md` and `pack.yaml`
  **before** the first calibration and never change it within a tag.
- CALIBRATE runs the baseline on all calibration seeds → `sigma`. The pack is **underpowered iff
  2 × expected holdout SE > MES** and then stops before spending the holdout.
  - seed packs: SE = √2·sigma/√n_holdout_seeds
  - item packs: SE = √(0.2/n_holdout_items + 2·sigma²/n_holdout_seeds) (or measured sd if a reference exists)
- **Size the eval so that MES ≥ 2.5 × expected SE.** For item packs with sigma≈0:
  n_holdout ≥ 0.2·(2.5/MES)²  → MES 0.05 needs 500 items, 0.07 needs 255, 0.10 needs 125, 0.15 needs 56.
- Seed choices:
  - training packs: `{calibration: [1,2,3,4,5], screen: 1, confirm: [2,3], holdout: [101,102,103]}`
  - deterministic eval-only packs (temperature 0): `{calibration: [1,2], screen: 1, confirm: [], holdout: [101]}`.
    `confirm: []` requires sigma ≤ 0.25·√(0.2/n_validation_items) (item packs) or sigma = 0 (seed packs),
    otherwise CALIBRATE stops with a config error.
- The first calibration seed must be the screen seed; screen and confirm seeds must be calibration seeds;
  holdout seeds are disjoint.

## 4. Budgets and speed

The frozen harness owns whatever the budget counts. Training packs: the harness runs the batch loop and stops
at `budget.limit` (tokens/steps); the surface supplies model/optimizer/`train_step`. Eval-only packs: every model
call goes through the frozen budgeted client, and the runner measures `service_tokens` from vLLM's counters.
A run whose `budget.json` exceeds the limit is `invalid`. Aim for **≤ 10 min per experiment** (RUN + EVALUATE)
on an idle GB10; `run.timeout_s` is a hard cap (docker kill → `timeout`). Add a `train_s` ratio guard (1.3×)
so "more compute per token" cannot win.

## 5. Patterns

**Training pack (token budget).** Use `arlab.lib.lm.token_budget_loop(surface, tokens_path, budget, batch,
seq_len, seed, out, config)`; the surface exposes `build(config)`, `train_step(state, (x, y), step,
total_steps) -> loss`, `save(state, path)`, `load(path, device) -> model`. Score with `arlab.lib.lm.score_bpb`
(cross-entropy from the model's logits, never the surface's loss) and `causality_check`. See `nanochat-lite`.

**LoRA fine-tune of a cached HF model.** Same shape: harness loads the base model from `/hf` (offline),
the surface builds the `peft` config/optimizer and `train_step`; the evaluator merges/loads the adapter and scores
generations or log-likelihoods itself. Put `peft` in `requirements.txt`.

**Eval-only pack with a model service.** Declare a vLLM service in `pack.yaml` (see `memory-longmemeval`):
```yaml
services: [{name: llm, image: "nvcr.io/nvidia/vllm:26.04-py3", command: "vllm serve Qwen/Qwen3.5-4B --port 8000",
            port: 8000, health_url: /health, mem_gb: 30, max_model_len: 32768, gpu: true, env: {}, volumes: []}]
budget: {unit: service_tokens, limit: N}
```
The runner starts it once per campaign on a private network (hostname = service name), adds
`--gpu-memory-utilization mem_gb/119` and `--max-model-len`, and counts prompt+generation tokens around RUN.
The surface gets a budgeted client from the harness (`arlab.lib.client`) — never an URL. Use temperature 0
and the deterministic seed defaults.

**Wrapping an external repo.** Copy a pinned snapshot during PREPARE (`git -C <repo> archive <sha> | tar -x`
into `/data/...`), pin the SHA inside `prepare.py` so it is part of `data_hash`, never modify the repo.

**Coding tasks with hidden tests.** Use `arlab.lib.cleanroom.run_hidden_tests` in EVALUATE: it copies only the
declared source files and the hidden tests into a fresh dir, runs pytest as uid 65534 with `-p no:cacheprovider
--noconftest`, and scores 1 only if the parent-read junit report lists exactly the expected test IDs, all passed.

## 6. Deterministic scorers

Exact-match after a frozen normalization (lowercase, strip punctuation/articles, number words → digits) against
the gold and frozen aliases; no LLM judge (answers could prompt-inject it); hedged or multi-candidate answers score 0.
Unit-test the scorer in `tests/` on hand-written ambiguous answers.

## 7. What `arlab check` enforces

- `--static`: required files; `pack.yaml` schema (unknown fields, seeds, guards, references under `frozen/run/`,
  `agent.visible` only `program.md`/`IDEA.md`/`frozen/run/*`); name = folder; non-empty surface; surface compiles;
  image builds; PREPARE succeeds (or reuses the data dir); pack tests pass.
- full: plus a PROBE — the baseline on the screen seed through RUN + EVALUATE with the GPU lock, a valid
  `metrics.json`, the budget within its limit, and each service's measured GPU memory ≤ 110 % of `mem_gb`.

## 8. Checklist before `arlab run`

- [ ] `IDEA.md` states hypothesis, MES (fixed), budget; `pack.yaml` matches.
- [ ] Eval sized so MES ≥ 2.5 × expected holdout SE; item packs write `splits.json`.
- [ ] Evaluator scores raw outputs only; emits `items` for per-item metrics; rejects NaN/shape errors as invalid.
- [ ] Tests cover the scorer and at least one anti-cheat case relevant to the pack.
- [ ] Independent astra review done (docs/ORCHESTRATOR.md step 3); findings fixed or recorded in DECISIONS.md.
- [ ] Full `arlab check` passes.
