ACTIVE — build (M0–M6) is DONE (all accept targets pass, see below); now working the owner's follow-up requests.

## Owner follow-ups (2026-09-28)
1. agentic-coding-small — root cause found (logged pilot, _pilots/agentic-mtp): 98.9% of task time is model latency;
   55% of tasks exhaust 30 steps; time-outs were format-error loops (not counted as steps); two waves of 20 made the
   slow tail set RUN time. Fixes (732bd8c): MTP-2 serving (1.56×), 40-model-call cap, one 40-task wave, campaign
   sized for ≥ 10 experiments. MTP-only pilot: RUN 2593 s (was ~4000), timeout 0.10 (was 0.225), pass 0.20.
   Campaign tag m6 queued (arlab-next) after the pilots.
2. TESTS step bug fixed (8c9acb1): pack tests never ran since M1; all 8 packs now pass their tests via arlab check.
3. New packs built + committed: stencil-focus, looped-latent, ttt-context (TTT stage 0), plastic-agent (stage 1).
   GPU pilots queued (arlab-gpu-queue: looped B0 done=0.0, B1, K4, stencil headroom; arlab-ttt-pilot-timing;
   arlab-next: plastic gate). Outputs in ~/arlab-runs/_pilots/.
4. Research briefs: looped-latent, memory-architecture (TTT merged into plastic-agent track), plastic-agent,
   TTT-DEEP-DIVE (MindsAI).
Next: read pilots → calibrate constants (looped difficulty, stencil go/no-go, ttt context length, plastic gate) →
astra review → full check → campaigns after m6.

# STATUS (build)

| Target | Result |
|---|---|
| accept-M0 | PASS (GPU sm_121, vLLM 26.04, 3 containerized codex calls, owner login untouched) |
| accept-M1 | PASS (32 tests: unit + CPU fixture campaigns, every status and verdict, kill -9 resume, tamper refusal, clean room) |
| accept-M2 | PASS (nanochat-lite: pilot, scripted statuses, anti-cheat b/c/e/f/g) |
| accept-M3 | PASS (nanochat-lite/m3b: 28 Codex experiments, 9.6/h, verdict supported; kill -9 during agent + training resumed) |
| accept-M4 | PASS (digits-label-smoothing via docs/ORCHESTRATOR.md; astra review; verdict not_found_at_this_scale; arlab/ unchanged) |
| accept-M5-memory-longmemeval | PASS (Qwen3-1.7B; 25 candidates; verdict not_found_at_this_scale) |
| accept-M5-agentic-coding-small | PASS (80 astra tasks at the model's edge; m5c: 3 candidates; verdict inconclusive: stopped_early:max_hours) |
| accept-M6 | PASS (zero tracebacks, all finalized by their own rules; 7.82 active h in M3 + M5) |

stencil-focus: awaiting the owner's ideas/stencil-focus/IDEA.md (not a failure per PLAN §5 M5).
BLOCKED.md: B1 (containerized Codex) resolved by the owner on 2026-09-26.
Docs: README.md (results), docs/retro.md, docs/ORCHESTRATOR.md, docs/PACK-AUTHORING.md, DECISIONS.md, WORKLOG.md.
No background jobs running.

Next (owner's choice): write ideas/stencil-focus/IDEA.md, or run new campaigns per docs/ORCHESTRATOR.md
(e.g. agentic-coding-small with a wall-clock limit sized from m5c, see docs/retro.md).
