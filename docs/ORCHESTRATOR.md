# ORCHESTRATOR — the Claude Code playbook for running arlab

You are Claude Code, run **interactively** by the owner on the Spark (their subscription). You turn prose ideas
into packs, launch and monitor campaigns, handle pauses and blocks, read reports and call Codex for reviews.
The arlab runner (a `systemd --user` unit) does the experiments; Codex (GPT-6, containerized) is the only agent
arlab calls. Read `CLAUDE.md` (hard prohibitions) first; they always apply.

## The workflow: "build and run the idea in `<file>`"

1. **Scaffold** (no LLM): `~/arlab/.venv/bin/arlab new <name> --idea <file>` → `ideas/<name>/` from
   `templates/pack/`, with the prose copied into `IDEA.md`. Names: lowercase, digits, `-`/`_`.
2. **Author the pack** following `docs/PACK-AUTHORING.md`: `pack.yaml`, `frozen/run/prepare.py`,
   `frozen/run/harness.py`, `frozen/eval/evaluate.py`, the baseline `surface/`, `program.md`, `tests/`.
   Reuse `arlab/lib` helpers (`lm`, `client`, `cleanroom`). Size the eval for the power check and write the
   MES into `IDEA.md` and `pack.yaml` **before** any calibration. Iterate with
   `arlab check --static ideas/<name>` (no GPU). If PREPARE may take > 5 min, run the check detached (below).
   External repos: pinned snapshot via `git -C <repo> archive <sha> | tar -x` inside PREPARE; never modify them.
3. **Independent review** (every new pack): one read-only `gpt-6-astra` call in the arlab agent container,
   pack folder mounted read-only, no data:
   ```bash
   P=~/arlab/ideas/<name>; R=$(mktemp -d /tmp/arlab-review-XXXX)
   cat > $R/prompt.md <<'EOF'
   You are reviewing an experiment pack for an automated research loop. /pack is read-only.
   An agent will edit only /pack/surface; /pack/frozen/run is the RUN harness it cannot change;
   /pack/frozen/eval is the hidden evaluator. Find: evaluator gaming (ways the surface can raise the score
   without the intended improvement), leakage between train/validation/holdout or from private data into RUN,
   budget loopholes (compute or model tokens not counted), and scorer bugs. List concrete findings with file:line
   and a one-line fix each; say "none" if a category is clean. Do not modify anything.
   EOF
   docker run --rm -i --name arlab-review-<name> --user 1000:1000 -e HOME=/tmp \
     -v ~/.cache/arlab/codex-home:/codex -e CODEX_HOME=/codex -v $P:/pack:ro -v $R:/out \
     arlab-agent:0.157.1 codex exec --ephemeral -C /pack --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox \
       -m gpt-6-astra -c model_reasoning_effort='"high"' -o /out/review.md - < $R/prompt.md > $R/events.jsonl 2>$R/stderr.log
   cat $R/review.md
   ```
   Fix real findings; record the rest (with why) in `DECISIONS.md`.
4. **Check**: the full `arlab check ideas/<name>` must pass (PROBE uses the GPU lock if the pack needs the GPU).
   Fix and repeat. After 3 failed rounds write the failing checks into `ideas/<name>/CHECK-FAILED.md`, tell the owner.
5. **Run and monitor**:
   ```bash
   ~/arlab/.venv/bin/arlab run ideas/<name> --tag <tag> --detach      # unit arlab-<name>-<tag>
   ~/arlab/.venv/bin/arlab status <name> --tag <tag>
   ```
   Arm a wake-up (the session does not notice the unit finishing):
   `while systemctl --user is-active -q arlab-<name>-<tag>; do sleep 300; done` as a background shell, or
   `/loop` at ≥ 20 min. When it finishes, read `~/arlab-runs/<name>/<tag>/report.md`, summarize it for the owner
   (verdict + rule, effect ± 2·SE, kept changes, speed, agent tokens) and suggest the next campaign.

Anything longer than 5 minutes that is not a campaign (e.g. a first `check` with a big PREPARE):
`systemd-run --user --unit=arlab-<name>-check --collect -p WorkingDirectory=$HOME/arlab
 -p StandardOutput=append:<log> -p StandardError=append:<log> $HOME/arlab/.venv/bin/arlab check $HOME/arlab/ideas/<name>`.
Never hold a tool call open for more than 10 minutes.

## Reading `arlab status`

| Field | Meaning |
|---|---|
| `phase` | sealed → probe → calibrate → loop → finalize → finalized |
| `unit` | systemd state; `failed` means the start limit (5 crashes/h) was hit — read `runner.log` |
| `waiting` | "waiting for the arlab GPU lock" (another GPU campaign) or "waiting for GPU/memory since … blocked by …" (owner's jobs, reservations, MemAvailable) — nothing to do, it polls every 60 s |
| `paused` | `codex auth` — see below |
| `baseline` / `incumbent` | screen-seed primary of the baseline and the current incumbent (keep id) |
| `sigma`, `expected_holdout_se`, `underpowered` | calibration noise and the power check |
| `accept_rate`, `per_hour_active` | keeps / experiments; experiments per active hour (GPU waits excluded) |
| `limits_left` | experiments, active hours, agent calls before the campaign stops itself |
| `last10` | id, status, primary, delta vs incumbent, description |

`results.tsv` has one row per experiment; `runs/<id>/` has `record.json`, `diff.patch`, logs, the agent view.

## Pause and stop reasons (and what to do)

| Situation | Action |
|---|---|
| `paused: codex auth` | arlab's Codex login expired. Ask the owner to run `CODEX_HOME=~/.cache/arlab/codex-home codex login --device-auth`; the runner re-checks every 10 min and resumes by itself. |
| stop `infra` (6 infra errors in a row) | Read the last `runs/*/agent/stderr.log` / `record.json` reasons (rate limit, network, docker). Fix the cause, then `arlab run … --tag <tag> --detach` again: it resumes (FINALIZE already ran, so a *new* tag is needed to continue searching). |
| stop `disk` | Free space by deleting arlab-created data only (discarded checkpoints, old `~/arlab-runs/*/*/runs/*/s*/out`, stale data dirs). Never touch other files. |
| stop `max_*`, `no_keep` | Normal end; read the report. |
| stop `stop` | Someone ran `arlab stop <name> --tag <tag>`. |
| verdict `inconclusive: underpowered` | The report states the measured sigma/SE. Enlarge the eval set or budget under a **new tag** (new data → new `data_hash`). Never change MES within a tag. |
| `config error: …` in report | The pack is broken (PREPARE/TESTS failed, baseline crashed, determinism check failed). Fix the pack and use a new tag. |
| unit `failed` / `TamperError` | Resume refused because `sealed/`, the data dir or the image changed. Investigate; start a new tag rather than "fixing" the sealed copy. |
| owner's GPU jobs appear | Nothing: runs overlapped by a foreign GPU process are re-run (`contended`), the runner waits for free memory. Record `waiting_gpu since <ts>` in STATUS.md if you are waiting on it. |

`arlab stop <name> --tag <tag>` finishes the current experiment, then FINALIZEs. `arlab report <name> --tag <tag>`
regenerates `report.md`. Re-running `arlab run` on a finalized tag only regenerates the report.

## Reading the report

- **supported** — holdout d ≥ MES and d − 2·SE > 0 (and every holdout seed positive for seed packs).
- **not_found_at_this_scale** — ≥ 10 experiments, powered, and the holdout upper bound d + 2·SE < MES:
  "the search found no effect ≥ MES in N experiments", never "no effect exists".
- **inconclusive** — `underpowered`, `holdout_uncertain` (interval straddles MES) or `stopped_early:<reason>`.
Keeps smaller than MES are fine (they accumulate); only the verdict is judged against MES.

## Hard rules (repeat)

No sudo. Never touch processes/containers/images/files arlab did not create. No pushing or publishing. Never run
Claude non-interactively (`claude -p`, Agent SDK, Anthropic API). Codex (containerized, arlab's `CODEX_HOME`)
is the only agent. Keep ≥ 60 GB free. Never end a turn with a campaign pending unless a wake-up is armed.

## Friction noted while running M4 (Appendix B, 2026-09-27)
- The review command drops Codex's stderr; add `2>$R/stderr.log` so a failed call can be diagnosed.
- The astra review flags by-design properties (the surface owns its optimizer; wall-time guards bound compute).
  Answer each finding in the pack's `REVIEW.md` (fixed / bounded by … / by design) so the next reader sees why.
- Public datasets bundled with the image (sklearn digits) let a surface look labels up: block the loader in the
  frozen harness and say so in `program.md`.
- On a seconds-long CPU pack, Codex latency dominates (agent share 93%, ~1 min/experiment); 20 experiments fit
  in 25 min, so `max_experiments` rather than `max_hours` ends such a campaign.
