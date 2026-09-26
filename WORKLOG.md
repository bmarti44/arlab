# WORKLOG (append-only)

- 2026-09-26 session start. Read PLAN.md, prior-art.md. Machine: 1.9T free, MemAvailable ~118 GB, no GPU compute apps, no arlab units.
- M0: CLAUDE.md + .claude/settings.json committed (2c84771).
- M0: pytorch:25.10 as uid 1000: torch 2.9.0a0 nv25.10, CUDA 13.0, cap (12,1), bf16 matmul + SDPA fwd/bwd OK.
- M0: vLLM 26.04 + Qwen3.5-4B @0.35 → completion OK (Qwen3-4B not actually cached). GPU PID→container via cgroup verified. GPU memory not in memcg.
- M0: arlab-agent:0.157.1 built. Containerized codex call DENIED by the auto-mode classifier → BLOCKED.md B1. M0 partial (codex part).
- M1: runner written (pack, stats, guards, execute, agent, record, campaign, experiment, report, cli). _fixture pack. `arlab check ideas/_fixture` PASS.
- M1: 14 unit tests pass. Manual scenario A: all statuses + anti-cheat (a)-(e) + supported verdict in 88 s.
