# STATUS

Milestone: M3 next (Codex campaign on nanochat-lite). Owner approved containerized Codex 2026-09-26 (B1 resolved).
- M0: accept-M0 PASS (GPU, vLLM, 3 codex calls, owner login untouched).
- M1: built; final `make accept-M1` running in the acceptance chain (unit arlab-accept-chain).
- M2: accept-M2 PASS earlier; re-running in the chain after the review fixes (stricter anti-cheat checks).
- Review: gpt-6-sol xhigh round 1 → 10 findings, all fixed (commit "Fix sol review findings"). Round 2 running
  (unit arlab-review-sol-2, /tmp/arlab-review-2/review.md).

Background jobs:
- arlab-accept-chain: accept-M0 → accept-M2 → accept-M1 (~/arlab-runs/_m1/{accept-m0,accept-m2,accept-m1}.log, chain.txt)
- arlab-review-sol-2: sol review round 2
- arlab-gen-p01: gpt-6-astra pilot of 10 agentic-coding tasks (~/arlab-data/agentic-gen/p01)

Next action: chain + review done → fix round-2 findings → launch M3 (`arlab run ideas/nanochat-lite --tag m3 --detach`,
max_hours 3) → during M3: validate pilot tasks, generate the rest (astra), M4 digits campaign (CPU) →
after M3: memory-longmemeval model selection/check/m5, agentic pilot/check/m5 → M6.

Packs: nanochat-lite (M2 done), memory-longmemeval (static check PASS; model selection pending),
agentic-coding-small (pack written; tasks being generated), digits-label-smoothing (M4; check PASS),
stencil-focus (awaiting owner IDEA.md).
