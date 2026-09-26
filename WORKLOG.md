# WORKLOG (append-only)

- 2026-09-26 session start. Read PLAN.md, prior-art.md. Machine: 1.9T free, MemAvailable ~118 GB, no GPU compute apps, no arlab units.
- M0: CLAUDE.md + .claude/settings.json committed (2c84771).
- M0: pytorch:25.10 as uid 1000: torch 2.9.0a0 nv25.10, CUDA 13.0, cap (12,1), bf16 matmul + SDPA fwd/bwd OK.
- M0: vLLM 26.04 + Qwen3.5-4B @0.35 → completion OK (Qwen3-4B not actually cached). GPU PID→container via cgroup verified. GPU memory not in memcg.
- M0: arlab-agent:0.157.1 built. Containerized codex call DENIED by the auto-mode classifier → BLOCKED.md B1. M0 partial (codex part).
- M1: runner written (pack, stats, guards, execute, agent, record, campaign, experiment, report, cli). _fixture pack. `arlab check ideas/_fixture` PASS.
- M1: 14 unit tests pass. Manual scenario A: all statuses + anti-cheat (a)-(e) + supported verdict in 88 s.
- M1: evaluate mount moved to /run_out (NVIDIA hook needs writable /run). trial() split into trial + evaluate_step.
- M1 tests: statuses/not_found/underpowered pass; uncertain fixed (MES 0.038); kill test flaked once (guard_fail) while nanochat PREPARE loaded the CPU → run accept-M1 in quiet periods.
- M2: nanochat-lite pack (port of autoresearch GPT + MuonAdamW, SDPA). PREPARE ~1 min (400M train tokens). PROBE: val_bpb 1.2204, RUN 228 s, EVAL 12 s, peak 9.5 GB, causal diff 0.
- M2: pilot calibration running (tag pilot). Scripted M2 campaign + (g) re-score check in scripts/accept_m2.py.
- M4 prep: templates/pack, docs/PACK-AUTHORING.md, docs/ORCHESTRATOR.md, README.md. M5 prep: arlab/lib/{client,cleanroom}.py; memory-longmemeval pack (static check PASS: 415 short-span questions → 207/207).
