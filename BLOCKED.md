# BLOCKED

## B1 — Containerized Codex call denied by Claude Code's auto-mode classifier (2026-09-26)
- **Check:** M0 "three containerized `codex exec` calls return schema-valid JSON" (and everything using Codex: M3, M4 astra review/agent experiments, M5).
- **Evidence:** the exact PLAN §3.6 `docker run … arlab-agent:0.157.1 codex exec … --dangerously-bypass-approvals-and-sandbox …` was refused by the auto-mode classifier ("Create Unsafe Agents"). The `.claude/settings.json` allow rule `Bash(docker:*)` did not override it.
- **Attempts:** none. The denial says not to work around it, so none were made; launching the runner with the Codex backend would be the same outcome and is also on hold.
- **Owner decision needed:** allow Claude Code to launch the containerized Codex agent (and the arlab runner that launches it). For example, approve it interactively, or add a permission rule you're comfortable with. Everything that doesn't need Codex (M0 GPU/vLLM parts, M1, M2) continues meanwhile.
