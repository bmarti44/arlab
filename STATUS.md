# STATUS

Milestone: M1 (runner core) — fixing fixture timing flakiness before the final `make accept-M1`.
M0: partial — every check except the three containerized Codex calls is done; Codex blocked (BLOCKED.md B1).
M2: nanochat-lite pack built, `arlab check` PASS; pilot calibration (tag pilot) finishing → then set MES, run the scripted M2 campaign (tag m2), then `make accept-M2`.

Background jobs:
- arlab-nanochat-lite-pilot (pilot calibration; ~/arlab-runs/nanochat-lite/pilot/runner.log)
- arlab-m1-uncertain2 (fixture holdout_uncertain scenario, to re-tune its MES after the batch change)
Waiting on: pilot results; owner decision on B1 (containerized Codex) for M0-codex, M3, M4, M5 agent campaigns.

Done ahead (CPU work, no Codex): docs/PACK-AUTHORING.md, docs/ORCHESTRATOR.md, README, templates/pack, `arlab new`,
memory-longmemeval pack (static check PASS), digits-label-smoothing (Appendix B) pack (full check PASS), stencil-focus README.
agentic-coding-small: needs gpt-6-astra-generated tasks (source 3) → waits on B1.

Next action: set fixture MES from uncertain2 → full `make accept-M1` (quiet box) → M2 scripted campaign → accept-M2 →
memory-longmemeval model selection + full check + calibration → then stop and report to the owner (B1).
