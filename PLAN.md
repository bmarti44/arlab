# PLAN.md — `arlab`: a small, honest autoresearch lab for one DGX Spark

**Status:** hand-off plan · **Date:** 2026-09-26 · **Owner:** Brian (bmarti44) · **Executor and orchestrator:** Claude Code, run interactively by the owner on the Spark with the owner's Claude subscription. It builds arlab, then operates it.
The prior-art survey is in `docs/prior-art.md`. It is reference only; this plan wins on any conflict.

---

## 0. Read this first (executor rules)

**What you are building:** a thin Python runner, `arlab`. It repeatedly asks a coding agent (Codex CLI, GPT-6) for *one* change to an editable file and runs that change under a fixed budget in a container. A frozen evaluator that the agent cannot see or touch scores the result, and the change is kept only if it beats measured noise. It works for **any research idea packaged as an "idea pack"** (a folder under `ideas/`). The owner will keep adding packs after you finish.

**Roles (fixed):**
- **Claude Code** (you: interactive, on the owner's subscription) builds arlab. Afterwards it is the **orchestrator**. It turns the owner's prose ideas into packs, launches and monitors campaigns, handles pauses and blocks, reads reports, and calls Codex for reviews.
  Claude Code is **never** run non-interactively. Nothing may run `claude -p`, the Claude Agent SDK or the Anthropic API, because that bills API usage instead of the subscription.
- **The arlab runner** (Python, a `systemd --user` job) owns the per-experiment loop, git, evaluation, statistics and the ledger. Campaigns keep running when the Claude Code session closes.
- **Codex CLI (GPT-6 family)** is the research agent inside the loop. It uses `gpt-6-sol` for routine proposals and `gpt-6-astra` for rare, high-stakes calls.

**The owner's priorities, in order:**
1. **Accurate:** no noise accepted as wins, no cheating, no silent bugs.
2. **Fast iteration.**
3. **Simple:** do NOT over-engineer. When in doubt, build less. §1.3 lists things you must not build.

**How to work (you may be restarted or compacted):**
- **First:** create `CLAUDE.md` at the repo root; it is loaded automatically in every session. It holds the roles above, the hard prohibitions below, "read STATUS.md first", the wake-up rule, and a pointer to `docs/ORCHESTRATOR.md` (written in M4). Also create `.claude/settings.json` with allow rules so auto mode never blocks the build: `Bash(docker:*)`, `Bash(systemd-run:*)`, `Bash(systemctl --user:*)`, `Bash(/home/bmarti44/arlab/.venv/bin/arlab:*)`, `Bash(make:*)`, `Bash(uv:*)`, `Bash(git:*)`, `Bash(nvidia-smi:*)`.
- **Tracking files at the repo root:**
  - `STATUS.md`: current milestone, running background jobs with unit names, what you are waiting on, next action. Rewrite it after every step.
  - `DECISIONS.md`: one line per decision, with the reason.
  - `WORKLOG.md`: append-only.
  - `BLOCKED.md`: only when needed.
  
  On any (re)start: read `STATUS.md`, then `systemctl --user list-units 'arlab-*'`, then continue.
- **Detached jobs:** anything that may take more than 5 minutes (training, campaigns, downloads, image builds, full pack checks) runs detached:
  `systemd-run --user --unit=arlab-<name> --collect -p StandardOutput=append:<log> -p StandardError=append:<log> <absolute command>`
  User units don't inherit your PATH, so use `~/arlab/.venv/bin/arlab` and absolute paths. Do CPU work meanwhile. Never hold one tool call open for more than 10 minutes.
- **Wake-up rule:** an interactive session does not notice a detached job finishing, so never end a turn with work pending unless a wake-up is armed. Use either a background shell wait (e.g. `while systemctl --user is-active -q arlab-<unit>; do sleep 300; done`) or a `/loop` wake-up at intervals of at least 20 minutes.
- **Timebox:** give each milestone 2× its estimate, counting only active work. Waiting for the owner's GPU jobs doesn't count; record it in `STATUS.md` as `waiting_gpu since <ts>`. If an acceptance check still fails after 3 genuinely different fix attempts:
  - add a `BLOCKED.md` entry (the check, the evidence, the 3 attempts, what the owner must decide);
  - mark the milestone `partial`;
  - continue with the next milestone that doesn't depend on it.
- **Acceptance:** every milestone ends with `make accept-M<n>`, a script whose exit code is the verdict. A milestone is done only when that script exits 0.
- **No questions mid-build:** don't stop to ask the owner. Choose the conservative option, record it in `DECISIONS.md`, and continue. When this plan and reality disagree, follow the plan's intent and record the deviation.

**Hard prohibitions (never, even to unblock yourself):**
1. No `sudo`. Nothing arlab does needs root on the host.
2. Never stop, kill, or reconfigure processes, containers, or services you did not start. The owner runs other GPU work, model servers, ComfyUI and other agent sessions on this box.
3. Never `docker system prune`, `docker image prune`, `docker builder prune`, or `docker rmi` anything not tagged `arlab-*`.
4. Never modify other repos or other projects' files. You may write only in these places:
   - `~/arlab`, `~/arlab-runs`, `~/arlab-data`
   - `~/.cache/arlab`, `~/.cache/uv`, `/tmp`
   - `~/.cache/huggingface` (downloads only, during PREPARE)
   
   `~/stencil-llm` is read-only.
5. Never push to any git remote, publish anything, or open issues/PRs. Local commits in `~/arlab` are fine; commit after each passing step.
6. Never run Claude non-interactively (see Roles). Inside arlab, the only agent is **Codex CLI with GPT-6 models**. No other paid APIs, no local LLMs for agent calls, no new Ollama pulls.
7. Keep at least 60 GB free on `/`. Delete only what arlab created, starting with checkpoints of discarded runs.

**Done means:** `STATUS.md` starts with `DONE` only if every required `accept-M*` target exits 0. Otherwise it starts with `INCOMPLETE` and lists each failing target and its `BLOCKED.md` entry. Either way, end with a summary to the owner.

---

## 1. Goal, scope, non-goals

### 1.1 Goal
One command runs an unattended, reproducible campaign for an idea pack. The campaign ends with a machine-written verdict (`supported` / `not_found_at_this_scale` / `inconclusive`), stating the rule that fired and the evidence.

A new idea is a new folder under `ideas/`, normally built by the Claude Code orchestrator from the owner's prose `IDEA.md`; the runner does not change. Packs never conflict with each other. Shared code lives in `arlab/lib/` and is imported, not copied.

### 1.2 Definition of done
| # | Goal | Proven by (scripted in `make accept-M<n>`) |
|---|---|---|
| D1 | Correct loop | CPU fixture campaigns (M1) reach every representative status and all three verdicts; anti-cheat tests pass |
| D2 | Fast | nanochat-lite reports median minutes per experiment and experiments/hour, excluding GPU waits. The target is ≥ 8/hour; it is reported, and missing it is labeled "speed target missed" rather than treated as a blocker |
| D3 | Unattended | the M3 and M5 detached campaigns each end by their own rules with zero runner exceptions; `kill -9` followed by a re-run resumes correctly |
| D4 | Honest | every pack gets calibration, fresh-seed confirmation (where §3.5 requires it), a one-shot holdout judged on measured uncertainty, and the anti-cheat suite |
| D5 | Generic | `nanochat-lite`, `memory-longmemeval`, `agentic-coding-small` and the Appendix B pack (built from prose via `docs/ORCHESTRATOR.md`) all run with the same commands and no runner changes. `stencil-focus` is built the same way once the owner has written its `IDEA.md` |
| D6 | Reproducible | each campaign records its sealed snapshot hash, data hash, image digest, `uv.lock` hash, git commits and codex version. The holdout retrains the winner from its commit on fresh seeds |

### 1.3 Scope guard — do NOT build these now
Do not build any of the following:
- **Search and tuning:** frontier/MAP-Elites/evolutionary search, Optuna/HPO, a literature phase, proposal pipelining (unless M3's measurement justifies it).
- **Agents and backends:** a reviewer/skeptic model inside the loop, an A/B backend mode, a model router, Claude/opencode/Ollama backends (keep a tiny backend interface instead), a local LLM for agent calls.
- **Statistics extras:** an adoption-level taxonomy, a "simplification keep" rule, periodic drift re-runs, a separate repro re-run, power-probability estimates, bitwise determinism as a requirement.
- **Isolation extras:** an AST scan of surfaces, a RUN-step sandbox, a separate grader image or compose stack, OpenShell.
- **Ops extras:** thermal stress tests, a GPU contention benchmark, writing to the machine-wide GPU reservation file (arlab only reads it), plots, HTML reports, Aim/MLflow, archives/backups, a GUI.
- **Outward-facing work:** upstream PRs or publishing.

Runner code target: ≈1,500 lines of Python plus tests. Anything not in this plan must be justified by a failure observed in a real campaign and recorded in `DECISIONS.md` first.

---

## 2. The machine (verified 2026-09-26) and the owner's pre-flight

### 2.1 Facts (re-verify in M0; trust measurements over this list)
- **Hardware and OS:** DGX Spark, GB10 (sm_121), 20-core ARM, **119 GB unified memory shared by CPU and GPU**, no swap. Ubuntu 24.04.4, DGX OS 7.2.3 (OTA 7.4.0), driver 580.159.03, CUDA 13.0.
- **Docker:** `docker run --gpus all …` works (NVIDIA Container Toolkit 1.19.1). The user is in the `docker` group. There is no passwordless sudo.
- **Local images to use** (pull nothing else unless unavoidable): `nvcr.io/nvidia/pytorch:25.10-py3`, `nvcr.io/nvidia/vllm:26.04-py3`, `nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04`, `node:22-bookworm`.
- **HF cache** includes, among others: Qwen3-0.6B/1.7B/4B, Qwen3-30B-A3B (BF16, ~61 GB), Qwen3.5-2B/4B, Qwen3.8-27B (+FP8/NVFP4), gpt2, SmolLM2-1.7B, bge-small-en-v1.5, `openai/gsm8k`.
- **Tools:** uv 0.11.7, Python 3.12.3, Node 22, git, tmux, jq, systemd 255 (`systemd --user` linger enabled). gh is authenticated; never push.
- **Codex:** `codex-cli 0.157.1`.
  - Verified flags: `exec -C`, `--skip-git-repo-check`, `--ephemeral`, `--output-schema`, `-o`, `--json`, `-` (prompt on stdin), `-c model_reasoning_effort=…`, `--dangerously-bypass-approvals-and-sandbox`, `login --device-auth`, `login status`.
  - GPT-6 models: `gpt-6-sol`, `gpt-6-astra`, `gpt-6-luna`.
  - `--output-schema` requires a strict schema: every property is required, and `additionalProperties: false`.
- **Claude Code:** 2.1.283. `--permission-mode` accepts `acceptEdits | auto | bypassPermissions | manual | dontAsk | plan`.
- **Memory on GB10:** `nvidia-smi --query-gpu=memory.*` returns `[N/A]`. Use:
  - free memory: `MemAvailable` from `/proc/meminfo` (it includes page cache, which is often large here);
  - other GPU users: `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.
  - `utilization.gpu` can stay stuck (e.g. 96% at 20 W with no processes). Never use it to decide whether the GPU is busy.
- **Docker cgroups** name containers by **ID** (`docker-<id>.scope`), not by name.
- **GPU reservations:** the GPU is shared with the owner's work. `~/stencil-llm/tools/gpu_reserve.sh` keeps a machine-wide file, `~/.gb10-gpu.reservations` (JSON lines `{"name","pid","started","eta_min","peak_gb"}`). arlab reads it to respect others' reserved memory. arlab never writes to it.
- **Claude auth:** `ANTHROPIC_API_KEY` is set in the environment with no credit. Start Claude Code with it unset so the subscription is used.

### 2.2 Owner pre-flight (the human, before hand-off)
1. **Free disk:** `df -h /` must show **≥ 300 GB available**. This is already true: removing the R2E-Gym images freed ~1.6 TB.
2. **Stop DeepSeek:** stop the DeepSeek work and its restart guard. This is the owner's own action, e.g. `sudo systemctl disable --now dsv4-guard.timer`; arlab itself never uses sudo.
3. **GPU jobs:** let running GPU jobs finish, or accept that arlab waits for them.
4. **arlab's own Codex login**, so its token refreshes never interfere with yours:
   `mkdir -p ~/.cache/arlab/codex-home && CODEX_HOME=~/.cache/arlab/codex-home codex login --device-auth`
   Then check that `CODEX_HOME=~/.cache/arlab/codex-home codex exec -m gpt-6-sol "say ok"` answers, and confirm the ChatGPT plan quota can sustain ~10 calls/hour for days.
5. **Start the executor** in tmux: `cd ~/arlab && env -u ANTHROPIC_API_KEY claude --permission-mode auto`.
   Paste the Appendix C kickoff, then run `/loop Continue per STATUS.md; stop looping when STATUS.md starts with DONE or INCOMPLETE`.
   After a restart or a subscription usage-limit reset, type "Continue per STATUS.md". Detached campaigns keep running meanwhile.
6. **Optional hygiene:** `chmod 600 ~/civitai_key ~/ollama_key`.
7. **Optional, any time:** write `ideas/stencil-focus/IDEA.md` (§6.4).

---

## 3. Design

### 3.1 Layout
```
~/arlab/                      # this repo (MIT). Code + pack definitions. Campaigns never modify it.
  pyproject.toml uv.lock Makefile README.md CLAUDE.md .claude/settings.json PLAN.md STATUS.md DECISIONS.md WORKLOG.md
  arlab/
    cli.py        # new | check | run | status | stop | report
    pack.py       # pack.yaml schema (pydantic), validation, sealing
    campaign.py   # lifecycle, resume, stopping rules, finalization, verdict
    experiment.py # one experiment: propose → apply → run → evaluate → decide → record
    execute.py    # docker run wrappers (prepare/run/eval/services), timeouts, kill, telemetry
    agent.py      # Backend protocol + CodexBackend (containerized codex exec)
    stats.py      # §3.5 and nothing else
    guards.py     # GPU/memory/disk gates, campaign GPU lock, reservations-file reader
    record.py     # record.json, results.tsv regeneration, constraints.md
    report.py     # report.md
    proposal.schema.json
    lib/          # shared helpers packs import: token-budget train loop, LM cross-entropy scorer + causality
                  # check, budgeted model client, clean-room pytest runner (uid 65534), vLLM service config
  docker/Dockerfile.agent     # FROM node:22-bookworm; npm i -g @openai/codex@0.157.1
  templates/pack/             # `arlab new` scaffold
  ideas/<name>/               # one folder per idea (§3.2)
  tests/                      # unit tests + CPU fixture campaigns; GPU tests marked `spark`
  docs/  ORCHESTRATOR.md  PACK-AUTHORING.md  spark-notes.md  prior-art.md
~/arlab-data/<name>/<data_hash>/  # immutable prepared data (+ MANIFEST.json)
~/arlab-runs/<name>/<tag>/    # one campaign: sealed/, state.json, work/ (own git repo), runs/<id>/, results.tsv, report.md, runner.log
~/.cache/arlab/               # campaign GPU lock, codex-home (arlab's own Codex login), compile caches
```

### 3.2 The idea pack
```
ideas/<name>/
  IDEA.md          # prose pre-registration (hypothesis, why, MES rationale). Not machine-read.
  pack.yaml        # the ONLY machine-read config (schema below)
  program.md       # instructions for the research agent (Appendix A)
  surface/         # the editable files — the only thing the agent may change
  frozen/run/      # prepare.py, harness.py (the RUN entry point), loaders, budgeted client
  frozen/eval/     # evaluate.py, scorers — mounted ONLY into EVALUATE
  requirements.txt # extra pip deps for the pack image (optional)
  tests/           # the pack's deterministic checks (run by `arlab check`)
```
Rules:
- A pack never reads or writes another pack's files.
- Shared code moves into `arlab/lib/` only when a *second* pack needs it.
- Campaigns run from a **sealed snapshot**, so editing the repo (for example `arlab/lib`) while a campaign runs never affects it.
- **Budget ownership:** the frozen harness owns whatever the budget counts. For training packs, the harness owns the batch loop and stops at the token budget; the surface supplies the model, the optimizer and a `train_step`. For eval-only packs, all model calls go through the frozen budgeted client, and the runner counts service tokens. The surface never reads argv, environment variables or data paths.

**Data.** Prepared data is immutable and lives in `~/arlab-data/<name>/<data_hash>/`.
- `data_hash` covers the prepare command, `frozen/run/`, `requirements.txt` and the resolved pack-image digest. Changed inputs produce a new directory, so a campaign's data never changes underneath it.
- The runner writes `MANIFEST.json` (content hashes) and sets every `private/` to mode `0700`.

```
train/                 training data (training packs)
validation/public/     inputs the RUN step may see (questions, task repos); may be empty
validation/private/    answers, hidden tests, eval tokens — EVALUATE only
holdout/public/        like validation/public, used only in FINALIZE
holdout/private/       like validation/private, used only in FINALIZE
```

**Isolation.** RUN never has `private/`, `/eval`, `/result` or (outside FINALIZE) the holdout mounted, so leakage from RUN is impossible. EVALUATE computes every score itself, from its own data, using raw outputs of the surface (logits, answers, workspaces); it never uses a number the surface computed. Surface code runs next to private data in two places: LM forward passes (accepted under the good-faith proposer; the frozen scorer and the causality check (f) apply) and the coding pack's hidden tests, which run in a clean temp dir as uid 65534, through `arlab.lib`'s clean-room pytest runner (§6.3).

**Container mounts.** Every step is a fresh `docker run --rm --cidfile <run_dir>/<step>.cid --name arlab-<name>-<tag>-<step>-<id>`. The image `arlab-<name>:<hash>` is the base image plus `requirements.txt`. It is installed with a constraints file taken from the base image's `pip freeze`, so the base torch is never replaced.

| Step | User | Network | Mounts (RO unless noted) |
|---|---|---|---|
| PREPARE | 1000 | yes | `/frozen`=frozen/run · `/data`=new data dir **RW** · `/hf`=HF cache **RW** (`HF_HOME=/hf`) |
| RUN | 1000 | none, or the campaign network if the pack has services | `/work`=surface copy · `/frozen`=frozen/run · `/arlab_lib` · `/data/train` · `/data/public` (this split) · `/out` **RW** · `/hf` (`HF_HUB_OFFLINE=1`) · `/cache` **RW** (compile cache) |
| EVALUATE | 0 inside the container (needed for the uid-65534 test runner and `private/`); outputs are `chown`ed to 1000 | same as RUN | `/frozen` · `/eval`=frozen/eval · `/arlab_lib` · `/work` · `/run`=the run's out · `/data/public` · `/data/private` · `/result` **RW** · `/hf` · `/cache` **RW** = a separate `~/.cache/arlab/<name>/eval-cache`, never shared with RUN (root-owned compile-cache files would break RUN) |

`PYTHONPATH=/frozen:/arlab_lib`.

**`pack.yaml` schema (complete; fields not listed here do not exist):**
```yaml
name: nanochat-lite
image: {base: nvcr.io/nvidia/pytorch:25.10-py3}
prepare: {command: "python /frozen/prepare.py --out /data", timeout_s: 3600}
run:
  command: "python /frozen/harness.py --out /out --seed {seed} --split {split}"   # always a frozen entry point
  timeout_s: 600                       # hard wall-clock cap (runner: docker kill)
  gpu: true
  mem_gb: auto                         # auto = max(16, 1.5 × peak measured in the probe run)
evaluate:
  command: "python /eval/evaluate.py --run /run --out /result/metrics.json"
  timeout_s: 300
budget: {unit: tokens, limit: 31457280}  # owned by the frozen harness / measured service tokens (§3.7)
metric: {name: val_bpb, direction: minimize, mes: 0.01}   # rates are fractions (0.05 = 5 points); fixed BEFORE calibration
guards:                                # each: name + one of max | min | max_ratio_vs_baseline | min_ratio_vs_baseline
  - {name: train_s, max_ratio_vs_baseline: 1.3}
  - {name: params_m, max: 60}
seeds: {calibration: [1, 2, 3, 4, 5], screen: 1, confirm: [2, 3], holdout: [101, 102, 103]}
services: []    # e.g. [{name: llm, image: nvcr.io/nvidia/vllm:26.04-py3, command: "...", port: 8000, health_url: /health,
                #        mem_gb: 40, max_model_len: 32768, gpu: true, env: {}, volumes: []}]
references: []  # optional frozen alternative surfaces, e.g. [{name: off, path: frozen/run/ref_off}]
verdict: {compare_to: baseline}        # or the name of a reference: the comparison the verdict is about
agent:
  model: gpt-6-sol
  effort: high
  visible: [program.md, IDEA.md]       # extra read-only context; may list files under frozen/run/ only
  timeout_s: 900
campaign: {max_experiments: 200, max_hours: 24, max_agent_calls: 400, stop_after_no_keep: 40, rescue_after: 8}
acceptance: {allow_underpowered: false}
```

**Run contract.** The runner passes the seed and split to the frozen `harness.py`. The harness imports and drives the surface, then writes everything the evaluator needs (checkpoints, predictions, per-task workspaces) to `/out`, together with `/out/budget.json` (for example `{"tokens_served": N}`).

**Evaluate contract.** `evaluate.py` writes `/result/metrics.json` atomically:
```json
{"valid": true, "primary": 1.1523, "metrics": {"val_bpb": 1.1523, "params_m": 26.3}, "items": null, "message": "ok"}
```
- `valid: false` → `primary: null`.
- Packs whose metric is a mean over questions or tasks **must** fill `"items": {"<item_id>": score, …}` (score 0/1 or a float).
- The runner adds its own measurements before checking guards: `train_s`, `eval_s`, `peak_mem_gb`, `gpu_temp_max`, `service_tokens`.
- A run whose `budget.json` exceeds `budget.limit` is `invalid`.

### 3.3 Campaign lifecycle — `arlab run ideas/<name> --tag <tag> [--detach]`
One idempotent command; re-running it resumes. State is in `~/arlab-runs/<name>/<tag>/state.json`, written atomically (tmp + rename). The `runs/<id>/record.json` files are the source of truth.
```
CHECK → PREPARE → TESTS → SEAL → PROBE → CALIBRATE → LOOP{PROPOSE → APPLY → RUN → EVALUATE → DECIDE → RECORD} → FINALIZE → REPORT
```
- **CHECK** (static; no GPU, no agent): schema, required files, `py_compile` of the surface, image build.
- **PREPARE**: compute `data_hash`. If `~/arlab-data/<name>/<data_hash>/MANIFEST.json` exists, reuse that directory. Otherwise run `prepare.command` into a new directory, then write `MANIFEST.json` and set the permissions.
- **TESTS**: the pack's `tests/`, which may use the prepared data.
- **SEAL**: copy `frozen/`, `pack.yaml`, `program.md`, `IDEA.md`, `requirements.txt`, `surface/` and the used `arlab/lib` files into `sealed/`, and record the seal hash, `data_hash` and image digest. From here on the campaign uses only `sealed/`, that data directory and that image. Resume verifies all three against the stored values, not against the live repo.
- **PROBE**: run and evaluate the baseline on the first calibration seed. Before starting, wait for `MemAvailable ≥ (numeric run.mem_gb, or 40) + 8 GB`. PROBE validates `metrics.json` and measures `peak_mem_gb` (which sets `mem_gb: auto`) and the output size. It counts as calibration seed 1.
- **`arlab check ideas/<name>`**:
  - The default runs CHECK + PREPARE + TESTS + SEAL (into a temp dir) + PROBE and prints pass/fail. It queues for the GPU lock if the pack needs the GPU.
  - `--static` stops before PROBE and never needs the lock.
- **CALIBRATE** (no agent): run the baseline on the remaining calibration seeds, and every reference (if any) on the screen seed. Compute `sigma`, the determinism check and the power check (§3.5).
  - If the pack is underpowered and `allow_underpowered` is false, go straight to REPORT with `inconclusive: underpowered`. The holdout is not spent. The report gives the measured numbers and what would fix them: a larger eval set or budget, under a new tag. It never suggests changing MES.
  - If `seeds.confirm` is empty, the pack must be deterministic enough to skip confirmation: item packs need `sigma ≤ 0.25 · sqrt(0.2 / n_validation_items)`, and seed packs need `sigma = 0`. Otherwise stop with a config error.
- **LOOP**: decisions per §3.5, agent per §3.6.
  - Statuses: `keep`, `discard`, `crash`, `oom`, `timeout`, `invalid`, `guard_fail`, `no_op`, `skip`, `interrupted`, `contended`.
  - **Infrastructure failures** (docker errors, agent CLI errors, rate limits, network) are not experiments. They are recorded as `infra_error`, don't count toward limits or streaks, and are retried with backoff (1 → 30 min). After 6 in a row, stop with reason `infra`.
  - **Codex auth failures** (401 / login required) pause the campaign instead: status shows `paused: codex auth`, and the runner re-checks `codex login status` (with arlab's `CODEX_HOME`) every 10 min.
- **RECORD**: `runs/<id>/record.json` (atomic) holds:
  - id, parent commit, commit, seeds
  - per-seed metrics and items, deltas, SE
  - status, reason, hypothesis tag, description
  - agent model, tokens, timings (propose/run/eval/wait)
  
  `results.tsv` is regenerated from the records:
  `id  commit  status  primary  delta  se  hypothesis_tag  model  propose_s  run_s  eval_s  wait_s  peak_mem_gb  gpu_temp_max  description`
  Large outputs (checkpoints) of non-kept runs are deleted right after the decision; logs, diffs, metrics and records are kept.
- **Stopping** is checked after every record. Stop on:
  - `max_experiments`, `max_hours` (active time), `max_agent_calls`, `stop_after_no_keep`
  - `arlab stop` (a sentinel file; stops after the current experiment)
  - disk: less than 60 GB + 2× the probe's output size free
  - the infra streak
  
  Then FINALIZE. `arlab run` exits 0 on every planned stop, and non-zero only on an unexpected exception; systemd's start limit (§3.9) catches crash loops.
- **FINALIZE** runs once per tag. Each holdout seed is recorded as it completes, and a completed holdout seed is never re-run.
  1. Holdout: run and evaluate the incumbent and the `compare_to` surface on `seeds.holdout`. If `compare_to` is `baseline` and there were no keeps, skip this step: `d = 0`, and SE is the expected holdout SE.
  2. Verdict (§3.5).
  3. REPORT.
  
  A later `arlab run` on a finalized tag only regenerates the report.
- **Resume**:
  1. Take an exclusive `flock` on the campaign dir.
  2. `docker kill` leftover containers, using the IDs in `runs/*/*.cid`.
  3. Rebuild the incumbent, counters and next id **from the ordered records**, then rewrite `state.json`.
  4. Any run dir without a `record.json` becomes `interrupted` (not counted).
  5. Hard-reset `work/` to the incumbent commit and continue.
  
  Calibration and holdout resume per completed seed.

### 3.4 Git topology
`~/arlab-runs/<name>/<tag>/work/` is its own git repo, created from `sealed/surface/`; its first commit is the baseline. Each experiment commits on top of the incumbent. Discards hard-reset to the incumbent, and keeps are tagged `keep/<id>`. The runner never runs git in `~/arlab`.

### 3.5 Statistics and decisions (the complete rule set)
Rates (accuracy, pass rate) are fractions in [0, 1]. `sigma` is the SD of the calibration runs (0 if they are identical). Every comparison of candidate A vs B on the same seeds and split yields an improvement `d` (sign-adjusted, so positive is better) and a standard error `SE`:
- **Item packs** (`items` present): take the paired per-item difference, averaged over the seeds used. `d` is the mean over items, and
  `SE = sqrt( sd(item differences)² / n_items + 2·sigma² / n_seeds )`.
- **Seed packs** (no `items`): `d` is the mean over seeds of the per-seed differences, and `SE = sqrt(2)·sigma / sqrt(n_seeds)`.

Rules:
- **Power check (CALIBRATE; a planning gate).** Compute the expected holdout SE with the formulas above:
  - seed packs: from `sigma` and the number of holdout seeds;
  - item packs: from an assumed `sd = sqrt(0.2)` (20% discordant items) and `n_holdout_items`, or from the measured baseline-vs-reference SE if a reference exists.
  
  The pack is **underpowered iff `2 · expected SE > mes`**. MES is fixed in `pack.yaml` before calibration and never changed within a tag. PACK-AUTHORING recommends sizing packs so that `mes ≥ 2.5 · expected SE`.
- **Screen** (the candidate on the screen seed vs the incumbent's value on that seed). This is a cheap filter, and every guard must pass.
  - With confirm seeds: pass iff `d > 1·SE`.
  - With `confirm: []` (deterministic packs): `keep` iff `d > 2·SE`. These validation keeps are provisional; the holdout decides.
- **Confirm** (the candidate on the **confirm seeds only**, because the screen seed was selected for being high). `keep` iff `d_confirm > 2·SE_confirm` and, for seed packs, every per-seed difference is `> 0`. Otherwise `discard`.
- **On keep:** the candidate becomes the incumbent, with its per-seed values stored for later comparisons. Keeps may be smaller than MES; small real gains accumulate. MES is judged only in the verdict.
- **Contended runs are never used for a decision.** A run during which a foreign GPU process appeared is re-run once after the GPU is free again. If it is contended again:
  - in the LOOP, it is recorded as `contended` and doesn't count;
  - in CALIBRATE and FINALIZE, the runner keeps waiting and retrying, because those numbers must be clean.
- **Verdict** (FINALIZE: the incumbent vs `verdict.compare_to` on the holdout, using the **observed** SE). Apply in order:
  1. `supported` — `d ≥ mes` **and** `d − 2·SE > 0` **and** (for seed packs with ≥ 2 holdout seeds) every holdout seed is positive.
  2. `not_found_at_this_scale` — not underpowered, ≥ 10 experiments ran, the stop reason is not `infra`, disk or `arlab stop`, **and** the holdout upper bound `d + 2·SE < mes`. The report words this as "the search found no effect ≥ MES in N experiments", not as proof that no effect exists.
  3. `inconclusive` — everything else, always with a reason: `underpowered`, `holdout_uncertain` (the interval straddles MES), or `stopped_early:<reason>` (fewer than 10 experiments).
  
  The report also gives the incumbent vs the baseline on validation (from the records) and the references' screen-seed results from CALIBRATE.

### 3.6 The research agent (containerized Codex, one experiment per call)
- **View.** For each call the runner creates `runs/<id>/view/` with:
  - the incumbent's surface files (editable);
  - copies of the `agent.visible` files;
  - `notes.md`, the agent's scratchpad, persisted across calls (the runner keeps the newest 8 KB);
  - `constraints.md`;
  - `history.md`, generated by the runner: calibration numbers, incumbent vs baseline, the last 30 records, and hypothesis tags tried with counts and outcomes;
  - `incumbent.diff`;
  - `prompt.md` (program.md plus the above).
  
  **Never** in the view: `frozen/eval/`, any data, other packs, or anything else from `$HOME`.
- **Call.** The container is the sandbox. arlab uses its own Codex login in `~/.cache/arlab/codex-home`, owned by uid 1000.
  ```bash
  docker run --rm -i --name arlab-<name>-<tag>-agent-<id> --user 1000:1000 -e HOME=/tmp \
    -v ~/.cache/arlab/codex-home:/codex -e CODEX_HOME=/codex \
    -v <run_dir>/view:/work -v <run_dir>/agent:/out -v ~/arlab/arlab/proposal.schema.json:/schema.json:ro \
    arlab-agent:0.157.1 codex exec --ephemeral -C /work --skip-git-repo-check \
      --dangerously-bypass-approvals-and-sandbox -m gpt-6-sol -c model_reasoning_effort='"high"' \
      --output-schema /schema.json -o /out/proposal.json --json - \
    < <run_dir>/view/prompt.md > <run_dir>/agent/events.jsonl
  ```
  Use absolute host paths. There is **no host fallback**: if the containerized call can't be made to work, that is a `BLOCKED.md` item.
- **Output.** The agent edits files in `/work` directly and ends with JSON matching the strict `proposal.schema.json`: all fields required, `additionalProperties: false`.
  `{"action": "edit"|"skip", "description": "≤120 chars", "hypothesis_tag": "slug", "constraint_learned": string|null}`
  The runner then:
  1. copies back **only** the surface files;
  2. diffs them against the incumbent (other changes are ignored and logged);
  3. runs `python -m py_compile` on changed `.py` files;
  4. commits.
  
  An `edit` with an empty diff is `no_op`.
- **Crash fix.** On `crash` (not `oom`, `timeout` or `invalid`): one fix call with the same view plus the last 80 lines of the log, then one re-run of the screen. If it still crashes, the status is `crash`.
- **Plateau rescue.** After `rescue_after` consecutive non-keeps, one `gpt-6-astra` (high) call with the full ledger and all kept diffs writes a strategy note of at most 30 lines into `notes.md`. At most once per `rescue_after` experiments.
- **Constraints.** `constraints.md` lives in the campaign dir and is injected into every call. It holds terse facts, each with its run id, taken from log regexes (CUDA OOM with shapes, an unsupported kernel/op on sm_121, NaN at step N) and from `constraint_learned`. Deduplicated; at most 50 lines.
- **Usage.** Per-call tokens come from `events.jsonl` and are summed in the report. All calls, including fix and rescue calls, count toward `max_agent_calls`. There is no dollar accounting.
- **Backend.** A `Protocol` with `propose(view) -> Proposal` and `fix(view, log) -> Proposal`. `CodexBackend` is the only implementation.

### 3.7 Budget enforcement and anti-cheat
The proposer is a good-faith model making one edit per call. These controls guard against honest mistakes and drift, backed by the holdout:
1. **Frozen ownership of the budget.** The harness runs the training loop and stops at the token limit, or the budgeted client caps model calls. The runner measures `service_tokens` from vLLM's `/metrics` counters (`vllm:prompt_tokens_total`, `vllm:generation_tokens_total`) before and after RUN.
2. **Runner measurements:** the wall-clock `timeout_s` and the `train_s` ratio guard, which limits compute per token.
3. **Isolation and frozen scoring** (§3.2): no private data in RUN, no evaluator in the agent view, and scores computed only by frozen code.

`tests/anticheat/` holds one scripted surface per case, run on the CPU fixture (M1) or on the relevant pack (M2/M5). Each must end as stated:
- (a) prints a fake metric → ignored
- (b) opens a `private/`, holdout or `/eval` path from RUN → not mounted → `crash`
- (c) writes into `/frozen` → read-only → `crash`
- (d) NaN/inf/missing outputs → `invalid`
- (e) makes 2× more compute per step → `guard_fail` (`train_s`)
- (f) *LM packs:* a non-causal model → `invalid`. Perturbing tokens after position *t* changes the logits at positions ≤ *t*.
- (g) *LM packs:* re-scoring the **same checkpoint** with the surface's loss function altered (e.g. multiplied by 0.5) → bit-identical `val_bpb`, because the scorer computes cross-entropy from the logits itself
- (h) *memory pack:* hedged or candidate-listing answers → scored wrong (§6.2)
- (i) *coding pack:* a harness that plants `conftest.py`, `sitecustomize.py`, `.pth` files, or code that calls `os._exit(0)` at import → no gain (§6.3)

### 3.8 Sharing the Spark
- **GPU vs CPU packs.** A pack needs the GPU if `run.gpu` is true or any service has `gpu: true`. Packs that don't need it skip the GPU lock and the GPU wait, so they never queue behind a GPU campaign.
- **One GPU campaign at a time.** `~/.cache/arlab/gpu.lock` (`flock`) is held from PROBE through FINALIZE, including the lifetime of any services.
- **arlab's own processes** are GPU PIDs whose `/proc/<pid>/cgroup` contains a container ID from this campaign's `.cid` files. Any other GPU process is foreign.
- **Wait for free** before PROBE, before services start, and before each RUN/EVALUATE. Every condition must hold:
  - there are no foreign compute processes;
  - `MemAvailable ≥ need + others' unmaterialized reserved peaks + 8 GB`, where the reserved peaks come from `~/.gb10-gpu.reservations` as `max(peak_gb − current usage, 0)` per live pid.
  
  Poll every 60 s with no deadline (the owner's choice). `arlab status` shows "waiting for GPU since … blocked by …". Waiting time goes to `wait_s` and is excluded from speed metrics and active-hour limits.
- **During a run:** sample `nvidia-smi` and `MemAvailable` every 5 s into `runs/<id>/telemetry.jsonl`. If `MemAvailable` drops below 6 GB, kill the run (`oom`) to protect the owner's processes. Foreign processes appearing mid-run are handled by the contended rule in §3.5.
- **Services** start on a per-campaign docker network. For vLLM, the runner adds `--gpu-memory-utilization {mem_gb/119:.2f}` (the default of 0.9 would take ~107 GB of unified memory) and `--max-model-len {max_model_len}`. `check` fails if measured memory exceeds `mem_gb` by more than 10%.

### 3.9 Status, reports, detach
- **`arlab status [name --tag T] [--json]`** shows: phase, incumbent vs baseline, sigma/SE, the last 10 rows, accept rate, experiments/hour (active), waiting or paused reason, limits left, and whether the systemd unit has failed.
- **`report.md`** contains:
  - the verdict and the rule that fired
  - calibration, determinism and the power check
  - kept changes, with diffs and per-seed/holdout deltas ± 2·SE
  - discards grouped by hypothesis tag
  - constraints
  - the time breakdown (propose/run/eval/wait) and experiments/hour
  - agent calls and tokens
  - environment: sealed hash, data hash, image digest, codex version, commits
- **`--detach`** runs:
  `systemd-run --user --unit=arlab-<name>-<tag> --collect -p Restart=on-failure -p RestartSec=60 -p StartLimitIntervalSec=3600 -p StartLimitBurst=5 -p StandardOutput=append:<campaign>/runner.log -p StandardError=append:<campaign>/runner.log ~/arlab/.venv/bin/arlab run …`

---

## 4. Adding a new idea (the orchestrator workflow — the real product)
The owner opens Claude Code in `~/arlab` (interactive, on the subscription) and says something like "build and run the idea in `~/ideas/foo.md`". Claude Code follows `docs/ORCHESTRATOR.md`:
1. **Scaffold.** `arlab new <name>` (no LLM) scaffolds `ideas/<name>/` from `templates/pack/` and copies the owner's prose into `IDEA.md`.
2. **Author the pack**, following `docs/PACK-AUTHORING.md`:
   - Fill in `pack.yaml`, `frozen/run/prepare.py`, `frozen/run/harness.py`, `frozen/eval/evaluate.py`, the baseline `surface/`, `program.md` and `tests/`, reusing `arlab/lib/` helpers wherever they fit.
   - Size the eval so the power check passes, and fix MES before calibration.
   - Iterate with `arlab check --static`, which needs no GPU.
   - If the idea needs another repo (e.g. stencil-llm), copy a pinned snapshot with `git -C <repo> archive <sha> | tar -x` into the pack's data during PREPARE. Pin the SHA in `frozen/run/prepare.py` (so it is part of `data_hash`) and never modify that repo.
3. **Independent review** (every new pack). Make one read-only `gpt-6-astra` call in the arlab agent container, with the pack folder mounted read-only and no data. Ask it for evaluator gaming, leakage between splits, budget loopholes and scorer bugs. Fix real findings; record the rest in `DECISIONS.md`.
4. **Check.** The full `arlab check ideas/<name>` must pass. Fix and repeat. After 3 failed rounds, write the failing checks into `ideas/<name>/CHECK-FAILED.md` and tell the owner.
5. **Run and monitor.** `arlab run ideas/<name> --tag <tag> --detach`. Monitor with `arlab status`, using a background wait or `/loop` at intervals of at least 20 minutes; the campaign doesn't need the session to stay open. Handle `paused`, `BLOCKED` and a failed unit. When it finishes, summarize `report.md` for the owner and suggest the next campaign.

`docs/PACK-AUTHORING.md` is the key document. It covers:
- the §3.2 contracts, budget ownership and mounts;
- data-split rules;
- sizing the eval for the power check, with `mes ≥ 2.5·SE` recommended;
- seed choices, for example deterministic eval-only packs: `{calibration: [1, 2], screen: 1, confirm: [], holdout: [101]}`;
- budgets that fit ≤ 10 min per experiment;
- patterns:
  - training packs: a harness-owned token-budget loop, or LoRA fine-tuning of an HF-cache model with `peft`;
  - eval-only packs: a vLLM service plus the budgeted client;
  - wrapping an external repo;
- deterministic scorers;
- the checklist `arlab check` enforces.

`docs/ORCHESTRATOR.md` is the orchestrator's playbook. It covers:
- the workflow above;
- how to read `arlab status` and reports;
- how to handle every pause or stop reason;
- the wake-up rule and the hard prohibitions;
- how to call Codex (the exact container command) for reviews.

---

## 5. Milestones (in order; estimates = focused work; timebox = 2×, active time only)

**M0 — Bring-up (½ day).**
- Write `CLAUDE.md`, `.claude/settings.json` and `docs/spark-notes.md` (measured versions).
- Check that the base image `nvcr.io/nvidia/pytorch:25.10-py3` works as uid 1000: the GPU is visible, `torch.cuda.get_device_capability() == (12, 1)`, and bf16 matmul plus `F.scaled_dot_product_attention` fwd/bwd both run.
- Start `nvcr.io/nvidia/vllm:26.04-py3` once with Qwen3-4B at `--gpu-memory-utilization 0.35` and get one completion, while page cache is large as usual. If vLLM refuses to start for lack of free memory, write `BLOCKED.md`: the owner must decide how to free page cache, because arlab has no sudo.
- Build `arlab-agent:0.157.1`. Three containerized `codex exec` calls, using the exact §3.6 command with arlab's `CODEX_HOME`, must return schema-valid JSON. The owner's own `codex login status` must be unaffected.

`accept-M0 --cpu-only` may pass while the GPU is busy; the GPU part must pass before M2.

**M1 — Runner core on a CPU fixture, no LLM (1½ days).**
Build everything in §3.2–3.5, §3.7 and §3.8 plus FINALIZE, verdicts and the report. A `ScriptedBackend` drives it: it applies a list of prepared edits, indexed by the number of non-interrupted experiments.
`ideas/_fixture/` is a CPU-only pack (the pytorch image, `run.gpu: false`). It trains a tiny model in about 3 s with seed-dependent noise and a known better setting. One variant emits `items`, and a deterministic variant uses `confirm: []`.

`accept-M1`:
- Unit tests pass, including state reconstruction from records, the verdict ordering and the contended rule (simulated).
- Scripted campaigns reach `keep`, `discard`, `crash`, `timeout`, `invalid`, `guard_fail`, `no_op`, `skip`, `infra_error` and `interrupted`, plus every verdict: `supported`, `not_found_at_this_scale`, and `inconclusive` for both `underpowered` and `holdout_uncertain`.
- Anti-cheat cases (a)–(e) pass.
- `kill -9` of `arlab run` during RUN, and again between RECORD and the next PROPOSE, followed by `arlab run`, gives the same `(status, primary, hypothesis_tag)` sequence as an uninterrupted campaign after dropping `interrupted` rows.
- A tampered `sealed/` or data directory makes resume refuse.

**M2 — `nanochat-lite` on the GPU (1 day).** §6.1.

`accept-M2`:
- `arlab check` passes.
- A pilot calibration (tag `pilot`) sets MES and the token budget. This is allowed only for this lab-calibration pack; record why in `DECISIONS.md`. A fresh tag then calibrates as not underpowered.
- A scripted campaign (a known-good change such as a better learning rate, a crash, an OOM, a non-causal model) produces the expected statuses.
- Anti-cheat (b), (c), (e), (f) and (g) pass on this pack.

**M3 — Codex backend (1 day).** §3.6 in full.

`accept-M3` (campaign `max_hours: 3`):
- A detached nanochat campaign with `gpt-6-sol` runs ≥ 2 active hours with zero runner exceptions and ≥ 12 experiments.
- Every `edit` record has a diff and a description.
- The report shows the time breakdown and experiments/hour, and labels "speed target missed" if the rate is below 8/hour.
- `kill -9` during an agent call and during training, then resume, works.

Record the median minutes per experiment. **If** agent time is > 25% of the cycle and throughput is < 8/hour:
1. First try cheaper fixes (a smaller `history.md`, data-load caching).
2. Only if needed, implement one speculative proposal in flight while the GPU trains. It is discarded if the running experiment is kept, and it counts toward `max_agent_calls`.
3. Record the effect.

**M4 — Orchestrator workflow (1 day).** Build `arlab new` (scaffold only), `templates/pack/`, `docs/PACK-AUTHORING.md`, `docs/ORCHESTRATOR.md` and a README walkthrough, and polish `status`/`stop`/`report`/`--detach`.

`accept-M4`:
- Acting as the orchestrator and following `docs/ORCHESTRATOR.md` step by step, turn the prose idea in Appendix B into a pack.
- Get its astra review and pass `arlab check`.
- Complete a ≤ 30-minute campaign with ≥ 3 agent experiments recorded and a verdict.
- `arlab/` must not change during this step (`git diff --stat` on `arlab/`).

Record any friction in the docs themselves.

**M5 — Real packs (≈3 days; one at a time; each timeboxed at 1 day of active build work).** Build each pack through the §4 workflow. The GPU lock serializes their GPU work: while one pack's campaign runs detached, author the next with `check --static`.

For each required pack (`memory-longmemeval` §6.2, then `agentic-coding-small` §6.3), `accept-M5-<name>`:
- The full `arlab check` passes.
- Calibration is **not** underpowered. If it is, enlarge the eval set or budget under a new tag within the timebox; never change MES within a tag.
- At least 1 agent candidate is fully evaluated.
- The campaign (`max_hours: 6`, `stop_after_no_keep: 25`) ends by its own rules with a verdict and report. Any verdict counts.
- `accept-M5-memory-longmemeval` also shows that the resident vLLM service was not treated as foreign across ≥ 2 evaluations.

`stencil-focus` (§6.4) is built the same way only if the owner's `IDEA.md` exists by then. Otherwise it is listed as "awaiting owner IDEA.md", which is not a failure.

**M6 — Wrap-up.**
- `accept-M6` checks the `runner.log` of the M3 and M5 campaigns: zero tracebacks and every campaign finalized with a verdict by its own rules. It reports the total active hours but sets no minimum.
- Write `docs/retro.md`: where GPU time went, accept rates, agent latency and tokens, what broke.
- Update `README.md`.
- Set `STATUS.md` to `DONE` or `INCOMPLETE` per §0.

---

## 6. Packs

### 6.1 `nanochat-lite` (training; the lab's calibration pack)
- **Sources.** Port the model and training step from `karpathy/autoresearch` (MIT), using the GB10-friendly settings published by `mazar/autoresearch-spark`. That repo has no license file, so take numbers and ideas only, no code. Read both (≤ 30 min).
- **Settings.** Depth 4–6, vocab ≈ 4096–8192, seq len 512–1024, `F.scaled_dot_product_attention`, no FlashAttention-3.
- **Data.** Use upstream's data if its prepare works on arm64 within 30 min; otherwise use TinyStories. `train/` holds the shards; `validation/private/` and `holdout/private/` hold disjoint, fixed eval token sets.
- **RUN.** `frozen/run/harness.py` owns the loop. It loads fixed-size token batches from `train/` and calls the surface's `build(config)`, `train_step(state, batch, step, total_steps)` and `save(state, path)` until the token budget is spent, then writes `budget.json`. The surface controls the model, optimizer, schedule and any gradient accumulation. Choose the budget so the baseline trains in ≈ 4 min on an idle GB10.
- **EVALUATE.** `frozen/eval/evaluate.py` uses the `arlab.lib` LM scorer:
  - it loads the checkpoint with the surface's model class;
  - it feeds each eval sequence `tokens[:-1]` once;
  - it computes `log_softmax` and cross-entropy against `tokens[1:]` itself, never through surface code, and reports `val_bpb`;
  - it runs the causality check (f);
  - it reports `params_m`.
- **Metric and guards.** `val_bpb`, minimize. Guards: `train_s ≤ 1.3× baseline`, `params_m ≤ 60`, `peak_mem_gb ≤ 60`.

### 6.2 `memory-longmemeval` (eval-only; model server as a service)
- **Hypothesis.** A better memory subsystem (write/compress/retrieve/assemble) around a fixed small model improves long-conversation QA accuracy under a fixed per-question token budget.
- **Data.** LongMemEval `_s` (verify the current source; it has 500 questions).
  - PREPARE **drops question types whose gold answer is not a short span**, such as preference questions with rubric-style gold, and records which were dropped.
  - The remaining questions are split with a fixed seed, stratified by type, half validation and half holdout.
  - Set `mes` before calibration so that `mes ≥ 2.5·sqrt(0.2/n_holdout)`, rounded up to 0.01 (≈ 0.07 for 225 holdout questions). Record it in `IDEA.md`.
  - If a validation pass can't fit in 30 min, subsample both splits equally and recompute `mes`, also before calibration.
  - Use the deterministic-pack seed defaults.
- **Model.** Served from the HF cache (offline) by `nvcr.io/nvidia/vllm:26.04-py3` as a pack service, temperature 0. Pick the smallest cached Qwen3/Qwen3.5 model whose baseline accuracy lands between 0.2 and 0.8. Haystacks exceed the context window, so the baseline is retrieval-based: PREPARE may precompute `bge-small-en-v1.5` turn embeddings for a frozen retrieval helper.
- **Surface.** `surface/memory.py` implements `ingest(session)` and `answer(question) -> str`. The frozen harness calls these concurrently and routes every model call through `arlab.lib`'s budgeted client, which caps answers at 32 tokens.
- **Budget.** An **absolute** per-question token cap (`service_tokens ≤ N × n_questions`), fixed in `IDEA.md` and `pack.yaml` before calibration. Size N so one short LLM pass over each session fits; this lets write/compress designs compete with plain retrieval.
- **Evaluator.** Deterministic scoring only; there is no LLM judge, because answers could prompt-inject it. An item scores 1 iff the normalized answer (lowercase, punctuation stripped, number words mapped to digits, articles removed) **exactly equals** the normalized gold or one of the gold's frozen aliases, which PREPARE builds from the dataset's answer variants. Abstention items require an answer from the frozen abstention phrase set. Hedged answers ("Paris or London", "not Paris") therefore score 0. `evaluate.py` emits `items`, and `tests/` checks the scorer on hand-written ambiguous answers.
- **Metric.** `accuracy`, maximize.

### 6.3 `agentic-coding-small` (eval-only; long-horizon coding for small models)
- **Hypothesis.** For a fixed small local model, scaffold-level changes (planning, task tracking, self-verification, context management, tool design) raise the pass rate on multi-step coding tasks.
- **Surface.** `surface/agent.py` is a single-file agent harness in the mini-swe-agent style. It uses frozen tools (`read`, `write`, `edit`, `run(cmd)`, `finish`) inside a per-task directory, plus the budgeted model client (per-task step and token caps).
- **Tasks.** Self-contained, multi-step Python tasks that run in the pack image with no per-task Docker images. Each has hidden tests in `private/`, a list of declared source files, and the list of expected test IDs.
  - Sources, in order:
    1. a Terminal-Bench/Harbor subset needing only Python and common packages;
    2. task sets already on this machine from the owner's earlier experiments (look, don't ask; record what you found);
    3. tasks generated with `gpt-6-astra`, keeping only those whose hidden tests fail on the starter code and pass on a reference solution.
  - **Pilot first:** 10 tasks, timed, with test quality spot-checked.
  - Then build toward 60 validation / 60 holdout tasks. Set `mes` before calibration so that `mes ≥ 2.5·sqrt(0.2/n_holdout)`, rounded up to 0.05 (0.15 for 60), and record it in `IDEA.md`.
- **Model.** A cached model served via vLLM. Try Qwen3.5-4B, then Qwen3-4B, then Qwen3-30B-A3B last (BF16, ~61 GB; only if the small ones fall outside the band). Pick by measured speed and a baseline pass rate between 0.15 and 0.70. Run 4–8 tasks concurrently so that a validation pass takes ≤ 30 min.
- **Evaluator** (per task, via `arlab.lib`'s clean-room runner):
  1. The parent copies the task's declared source files and its hidden tests into a fresh temp dir readable by uid 65534.
  2. It runs `python -I -m pytest -p no:cacheprovider --noconftest --junitxml=<report dir>/junit.xml` there as uid 65534. The report dir is fresh, owned by uid 65534 and outside the task's temp dir; the parent reads it only after the child exits.
  3. The item scores 1 iff the exit code is 0 **and** the parent-read junit report lists exactly the expected test IDs, all passed.
  4. The temp dir is deleted afterwards.
  
  `evaluate.py` emits `items`.
- **Metric and guards.** `pass_rate`, maximize. Guards: `service_tokens` within the per-task cap; timeouts ≤ 10%.

### 6.4 `stencil-focus` (awaiting the owner's IDEA.md)
The owner will write `ideas/stencil-focus/IDEA.md`, describing the stencil-llm question to study. Until that file exists, create only `ideas/stencil-focus/README.md`, saying the pack is waiting for it. Once it exists, build the pack through the §4 workflow like any other idea.

Constraints that apply regardless:
- `~/stencil-llm` (GPL-2.0) is used only as a pinned snapshot copied during PREPARE, and it is never modified.
- arlab never evaluates on items stencil-llm treats as registered, SCREEN, TRAIN, final or holdout pools.
- stencil's FROZEN files are respected.
- The pack folder notes that it wraps GPL-2.0 code for local use.

---

## 7. Risks (and the answer in this plan)
| Risk | Answer |
|---|---|
| Noise mistaken for progress | calibration + power check, screen + fresh-seed confirm, item-level SE, contended runs excluded, one-shot holdout judged on observed SE, ordered verdict rules (§3.5) |
| Agent games the metric or drifts | view without eval or data, sealed snapshot + immutable data, frozen budget ownership, frozen scoring from raw outputs, clean-room hidden tests, anti-cheat suite, astra review of new packs (§3.2, §3.6–3.7, §4) |
| Box shared with owner's jobs | campaign GPU lock (GPU packs only), reads others' reservations, wait-for-free, memory watchdog, vLLM memory cap, never touches foreign processes (§3.8) |
| Codex quota, outages, auth | `infra_error` backoff; auth pause; arlab's own `CODEX_HOME` login (§3.3, §2.2) |
| Claude API charges | Claude Code runs interactively only, on the subscription; nothing runs `claude -p` or the Anthropic API (§0) |
| Runner crash / reboot | atomic records as the source of truth, idempotent `run`, flock, container-ID cleanup, systemd start limit (§3.3, §3.9) |
| Disk fills | 60 GB floor + output-size margin; delete non-kept checkpoints (§3.3) |
| Over-engineering | scope guard (§1.3), LOC target, "justify by observed failure" |
| Executor stalls | detached jobs, wake-up rule, `.claude/settings.json` allow rules, active-time timeboxes, `BLOCKED.md`, scripted accept targets (§0) |

---

## Appendix A — `program.md` template (≤ 80 lines; the runner appends `history.md` and `constraints.md`)
```
# <pack> — instructions for the research agent
You make ONE change per call. The runner (not you) runs training/evaluation, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator or the data; do not try.
## Goal
<maximize|minimize> <metric>. Current noise: SE ≈ <se>. A change is kept only if it beats the
incumbent on one seed and then clearly (by more than 2·SE) on fresh seeds.
## What you may change
<surface files>. Nothing else has any effect. Data and model calls come only through the frozen
harness/client; the surface must not read files, the environment or the network.
## What the code does (2–5 lines)
## Ideas worth trying (from IDEA.md)
## Rules
- One change, one hypothesis_tag (reuse an existing tag for the same idea).
- Read history.md: don't repeat a failed idea unless you change it materially. After 3 discards in a row, try something structurally different.
- Respect constraints.md (facts about this machine).
- Prefer simple changes; equal results with less code are better.
- Keep notes.md short and useful to your future self.
- Finish with the JSON object required by the output schema and nothing after it.
```

## Appendix B — Prose idea for the M4 new-idea test
> **IDEA: label smoothing for a small MLP on sklearn digits.** Hypothesis: label smoothing and small architecture/regularization changes improve the test accuracy of a 2-layer MLP trained for a fixed 1,500 optimizer steps (batch 64) on `sklearn.datasets.load_digits` (CPU only, a few seconds per run). Metric: accuracy, maximize; meaningful effect: 0.05 (5 points). Data: split the 1,797 digits 50/25/25 into train / validation / holdout with a fixed seed. Editable: the model, optimizer and training step. Budget: exactly 1,500 steps, run by the frozen harness.

## Appendix C — Kickoff message for the executor (paste as the first message)
> You are Claude Code, building `arlab` on this DGX Spark and later operating it as its orchestrator. Read `~/arlab/PLAN.md` in full, then `STATUS.md` if it exists. Follow §0 exactly:
> - Work milestone by milestone.
> - Run anything longer than 5 minutes detached, and never end a turn with work pending unless a wake-up is armed.
> - Keep `CLAUDE.md`, `STATUS.md`, `DECISIONS.md` and `WORKLOG.md` current.
> - Timebox, and write `BLOCKED.md` instead of stalling.
> - Never violate the hard prohibitions. In particular, never run Claude non-interactively; Codex is the only agent arlab calls.
>
> Build the smallest thing that meets the plan. Don't ask me questions; choose the conservative option, record it in `DECISIONS.md`, and continue. Stop when `STATUS.md` says `DONE` or `INCOMPLETE` per §0, and give me the summary.
