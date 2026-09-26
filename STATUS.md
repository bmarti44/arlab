# STATUS

Milestone: M2 (nanochat-lite on GPU) — scripted campaign tag m2 running detached; then `make accept-M2`.
M1: all scenarios pass individually; final full `make accept-M1` still to run in a quiet period (after m2).
M0: partial — every check except the three containerized Codex calls is done; Codex blocked (BLOCKED.md B1).

Background jobs:
- arlab-nanochat-lite-m2 (systemd --user; ~/arlab-runs/nanochat-lite/m2/runner.log). As of 19:33Z: calibration not
  underpowered (SE 0.0021 < MES/2); 0001 keep, 0002 crash, 0003 oom (as scripted), 0004 running.
- No orchestrator watcher: Claude Code reaped the session's wait shell under low system memory (19:31Z, during
  0004 training); not restarted pending the owner's go-ahead. The campaign itself is unaffected.
Waiting on: m2 finishing; owner decision on B1 (containerized Codex) for M0-codex, M3, M4, M5 agent campaigns.

Done ahead (CPU work, no Codex): docs/PACK-AUTHORING.md, docs/ORCHESTRATOR.md, README, templates/pack, `arlab new`,
memory-longmemeval pack (static check PASS), digits-label-smoothing (Appendix B) pack (full check PASS), stencil-focus README.
agentic-coding-small: needs gpt-6-astra-generated tasks (source 3) → waits on B1.

Next action: m2 finishes → `make accept-M2` → full `make accept-M1` (quiet box) →
memory-longmemeval model selection + full check + calibration → then stop and report to the owner (B1).
