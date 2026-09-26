# arlab — instructions for every Claude Code session

**Read `STATUS.md` first**, then run `systemctl --user list-units 'arlab-*'`, then continue.
`PLAN.md` is the spec (§0 = executor rules). After M4, `docs/ORCHESTRATOR.md` is the orchestrator playbook.

## Roles (fixed)
- **Claude Code** (interactive, owner's subscription) builds arlab, then orchestrates: turns prose ideas into packs,
  launches/monitors campaigns, handles pauses/blocks, reads reports, calls Codex for reviews.
- **arlab runner** (Python, `systemd --user` job) owns the experiment loop, git, evaluation, stats and the ledger.
- **Codex CLI (GPT-6)** is the only research agent: `gpt-6-sol` routine, `gpt-6-astra` rare/high-stakes.
  Always containerized (`arlab-agent:0.157.1`) with arlab's own `CODEX_HOME=~/.cache/arlab/codex-home`.

## Hard prohibitions (never, even to unblock yourself)
1. No `sudo`.
2. Never stop/kill/reconfigure processes, containers or services arlab did not start (owner runs GPU jobs, model servers, ComfyUI, other agents).
3. Never `docker system/image/builder prune`, never `docker rmi` anything not tagged `arlab-*`.
4. Write only in `~/arlab`, `~/arlab-runs`, `~/arlab-data`, `~/.cache/arlab`, `~/.cache/uv`, `/tmp`, and `~/.cache/huggingface` (downloads during PREPARE). `~/stencil-llm` is read-only. Never modify other repos.
5. Never push, publish, or open issues/PRs. Local commits in `~/arlab` only; commit after each passing step.
6. Never run Claude non-interactively (no `claude -p`, Agent SDK, Anthropic API). No other paid APIs, no local LLMs for agent calls, no Ollama pulls.
7. Keep ≥ 60 GB free on `/`. Delete only what arlab created.

## Working rules
- Anything > 5 min runs detached: `systemd-run --user --unit=arlab-<name> --collect -p StandardOutput=append:<log> -p StandardError=append:<log> <absolute cmd>` (use `~/arlab/.venv/bin/arlab`, absolute paths).
- **Wake-up rule:** never end a turn with work pending unless a wake-up is armed (background `while systemctl --user is-active -q arlab-<unit>; do sleep 300; done`, or `/loop` ≥ 20 min).
- Keep `STATUS.md` (rewrite each step), `DECISIONS.md` (one line per decision + reason), `WORKLOG.md` (append-only) current; `BLOCKED.md` only when needed.
- Milestones end only when `make accept-M<n>` exits 0. Timebox 2× estimate (active time). 3 failed fix attempts → BLOCKED.md, mark partial, move on.
- GPU busy with owner work → wait; record `waiting_gpu since <ts>` in STATUS.md. Use `MemAvailable` and `nvidia-smi --query-compute-apps`, never `utilization.gpu`.
- No questions to the owner mid-build: choose conservatively, record in DECISIONS.md.
- Scope guard PLAN.md §1.3 is binding. Runner ≈ 1,500 lines.
