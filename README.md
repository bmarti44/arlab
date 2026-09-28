# arlab — a small, honest autoresearch lab for one DGX Spark

arlab repeatedly asks a coding agent (Codex CLI, GPT-6, containerized) for **one** change to an editable
surface, runs it under a fixed budget in a container, scores it with a frozen evaluator the agent can't see,
and keeps it only if it beats measured noise. Every campaign ends with a machine-written verdict —
`supported`, `not_found_at_this_scale` or `inconclusive` — naming the rule that fired and the evidence.

Spec: `PLAN.md`. Pack authoring: `docs/PACK-AUTHORING.md`. Orchestrator playbook: `docs/ORCHESTRATOR.md`.
Machine notes: `docs/spark-notes.md`.

## Install

```bash
cd ~/arlab && uv sync                                  # creates .venv with the `arlab` CLI
docker build -t arlab-agent:0.157.1 -f docker/Dockerfile.agent docker   # the Codex agent image
mkdir -p ~/.cache/arlab/codex-home && CODEX_HOME=~/.cache/arlab/codex-home codex login --device-auth
```

## Walkthrough

```bash
arlab new my-idea --idea ~/ideas/my-idea.md   # scaffold ideas/my-idea/ from templates/pack/
# ... author pack.yaml, frozen/run/{prepare,harness}.py, frozen/eval/evaluate.py, surface/, program.md, tests/
arlab check --static ideas/my-idea            # schema, compile, image, PREPARE, pack tests (no GPU)
arlab check ideas/my-idea                     # + PROBE: the baseline through RUN + EVALUATE
arlab run ideas/my-idea --tag t1 --detach     # systemd --user unit arlab-my-idea-t1; survives the session
arlab status my-idea --tag t1                 # phase, incumbent vs baseline, sigma/SE, last 10, limits, waits
arlab stop my-idea --tag t1                   # stop after the current experiment, then FINALIZE
arlab report my-idea --tag t1                 # (re)write ~/arlab-runs/my-idea/t1/report.md
```

`arlab run` is idempotent: re-running it resumes from the records (`kill -9` safe). A campaign goes
CHECK → PREPARE → TESTS → SEAL → PROBE → CALIBRATE → LOOP → FINALIZE → REPORT:

- **SEAL** snapshots the pack and `arlab/lib` into `~/arlab-runs/<name>/<tag>/sealed/`; editing the repo never
  affects a running campaign, and resume refuses if the snapshot, the data or the image changed.
- **CALIBRATE** measures the baseline's seed noise (sigma) and the power check (2 × expected holdout SE ≤ MES).
- **LOOP**: propose (Codex in a container, sees only the surface + context) → apply (surface files only) →
  RUN → EVALUATE (frozen) → screen on one seed → confirm on fresh seeds → keep/discard → record.
- **FINALIZE**: the incumbent vs the baseline (or a reference) on held-out seeds and data, once; verdict rules in
  PLAN §3.5.

## Layout

```
arlab/            runner: cli, pack, campaign, experiment, execute, agent, stats, guards, record, report
arlab/lib/        shared helpers packs import: lm (token-budget loop, LM scorer, causality), client (budgeted
                  model client), cleanroom (hidden tests as uid 65534)
ideas/<name>/     one pack per idea (IDEA.md, pack.yaml, program.md, surface/, frozen/run, frozen/eval, tests/;
                  optional frozen/prepare/ = inputs only PREPARE sees, e.g. tasks with hidden tests; REVIEW.md)
templates/pack/   `arlab new` scaffold
tests/            unit tests + CPU fixture campaigns (`make accept-M1`)
~/arlab-data/<name>/<data_hash>/   immutable prepared data     ~/arlab-runs/<name>/<tag>/   campaigns
```

## Packs

| Pack | Kind | Metric |
|---|---|---|
| `_fixture` | CPU numpy MLP, scripted-backend tests | accuracy |
| `nanochat-lite` | **arlab self-test** (lab calibration): GPT pretraining under a token budget | val_bpb ↓ |
| `memory-longmemeval` | memory subsystem around a small vLLM-served model | accuracy |
| `agentic-coding-small` | scaffold for a small model on multi-step coding tasks | pass_rate |
| `digits-label-smoothing` | **arlab self-test** of the prose → pack workflow (`docs/ORCHESTRATOR.md`, M4) | accuracy |
| `stencil-focus` | awaiting the owner's `IDEA.md` | — |

## Results

| Campaign | Purpose | Verdict |
|---|---|---|
| `nanochat-lite/m3b` | arlab self-test | **supported** — optimizer batch 2^17 → 2^16 tokens: holdout val_bpb −0.0151 (d − 2·SE = 0.0102 > 0) |
| `digits-label-smoothing/m4` | arlab self-test | not_found_at_this_scale (effect < 0.026 < MES 0.05) |
| `memory-longmemeval/m5` | research | not_found_at_this_scale (25 candidates; effect < 0.062 < MES 0.08) |
| `agentic-coding-small/m5c` | research | inconclusive: stopped_early:max_hours (3 candidates in 6 h; ~70 min per validation pass) |
| `looped-latent/v0` | research | not_found_at_this_scale — looping layers 12–15 of Qwen3-0.6B (20 loop designs, loop-only surface, same training budget): holdout upper bound 0.025 < MES 0.03 |

The two self-tests check arlab, not research questions. **nanochat-lite** is the lab's calibration: it proves the
loop runs end to end on this GPU (speed, noise floor, kill -9 resume) and that the statistics keep one real change
while rejecting 27 near-misses. Its keep, a smaller optimizer batch under a fixed token budget, is the textbook
batch-size trade-off (and the edit M2's scripted test planted as "known good"), not a new finding. **digits** tests
that a prose idea becomes a working pack through the orchestrator. It was not designed as a negative control, but it
behaves as one: the baseline's 0.964 validation accuracy leaves < 3.6 points of headroom under a 5-point MES, so a
correct lab must return a null, and it did.

Per-campaign `report.md` files are in `~/arlab-runs/<pack>/<tag>/`; `docs/retro.md` covers time, accept rates,
agent latency/tokens and what broke. Tags that were re-run (m3 → m3b, m5 → m5c) and why: `docs/M3.json`,
`docs/M5.json`, `DECISIONS.md`.

## Acceptance

`make accept-M0 … accept-M6` — each target's exit code is the verdict (see `Makefile`, `scripts/`).
`make test` runs the fast unit tests.

## Rules that never bend

No sudo; never touch processes/containers/images/files arlab did not create; never push or publish; never run
Claude non-interactively (Codex is the only agent arlab calls); keep ≥ 60 GB free. See `CLAUDE.md`.

License: MIT. `ideas/nanochat-lite/surface/train.py` is ported from karpathy/autoresearch (MIT).
