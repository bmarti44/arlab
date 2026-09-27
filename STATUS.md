# STATUS

Milestone: M3 running (nanochat-lite/m3b, Codex gpt-6-sol, max_hours 3 active). Then M5 campaigns, then M6.
- M0 PASS, M1 PASS (31/31), M2 PASS, M4 PASS (digits-label-smoothing/m4: not_found_at_this_scale).
- M3: tag m3 calibrated anomalously (sigma 0.0185 → underpowered, no search; DECISIONS 2026-09-27); re-run m3b
  calibrated normally (sigma 0.0030, SE 0.0024 < MES/2). Loop started 03:16Z. Kill tests armed (unit arlab-m3-kills,
  ~/arlab-runs/_m1/m3-kills.log → m3b/kill-tests.json). Keep the box quiet (no GPU work, no heavy CPU) during M3.
- memory-longmemeval: Qwen3-1.7B; astra-review fixes changed data → full `arlab check` must be re-run after M3
  (then save as ~/arlab-runs/memory-longmemeval/m5/check.log) → m5 campaign.
- agentic-coding-small: 80 edge-level tasks (40/40), MES 0.20, prefix caching; astra-review fixes in → full
  `arlab check` after M3 (save as agentic-coding-small/m5/check.log) → m5 campaign.
- Sol reviews: 4 rounds done, all real findings fixed or recorded.

Background jobs: arlab-nanochat-lite-m3b (campaign), arlab-m3-kills (kill tests).

Next action: wait for m3b (≥ 2 active h, ~3 h) → `make accept-M3` → memory check + m5 (detached, 6 h) →
agentic check + m5 (6 h) → accept-M5-* → M6 (retro, README, accept-M6) → STATUS DONE/INCOMPLETE.
