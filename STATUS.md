# STATUS

Milestone: M3 next (nanochat-lite Codex campaign, max_hours 3), then M5 campaigns, then M6.
- M0 PASS, M1 PASS (31/31), M2 PASS, M4 PASS (digits-label-smoothing/m4: not_found_at_this_scale, 20 experiments).
- Sol xhigh review: 3 rounds, all real findings fixed or recorded (DECISIONS.md 2026-09-26/27).
- memory-longmemeval: model Qwen3-1.7B (baseline 0.291 on validation). Next: full `arlab check`, then m5 campaign.
- agentic-coding-small: difficulty ladder done (L1 4/5, L2 3/5, L3 3/5, L4 1/5, astra "p01" 0/10) → edge = L3–L4.
  7 astra batches (m01–m07: 3×L3, 4×L3b, 3×L4 each) generating; then validate, pick 40/40 (MES 0.20 by the
  subsample rule since 60/60 can't pass in 30 min), full check, m5 campaign. Testing vLLM prefix caching
  (hybrid Qwen3.5: off by default) for throughput.

Background jobs:
- arlab-gen-m01..m07 (astra task generation, ~/arlab-data/agentic-gen/m0*)
- prefix-cache A/B test (container arlab-pilot-pc; ~/arlab-data/agentic-pilot/pc-{off,on}.txt)

Next action: prefix-cache result → memory full check → launch M3 + m3_kill.py tests → validate tasks during M3 →
after M3: memory m5 → agentic check + m5 → accept-M3/M5 → M6.
