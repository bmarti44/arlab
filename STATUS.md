ACTIVE — build (M0–M6) is DONE (all accept targets pass, see below); now working the owner's follow-up requests.

## Owner follow-ups (2026-09-28)
| Pack | State |
|---|---|
| agentic-coding-small | root cause fixed (MTP, 40-call cap, one wave, ≥10 experiments); campaign **m6** queued (arlab-next, starts when arlab-p-* pilots finish) |
| looped-latent | **v0 done: not_found_at_this_scale** (upper bound 0.025 < MES 0.03, 20 loop designs) |
| stencil-focus | pilot GO at cap 2048 (oracle − baseline +0.269); full check running (arlab-p-stencil-check) |
| ttt-context (TTT stage 0) | 8K, no_ttt 0.18; astra fixes (trusted answering); re-pilot + full check running (arlab-p-ttt3) |
| plastic-agent (TTT stage 1) | astra fixes; gate pilot running (arlab-p-pa-gate): needs icl − none ≥ 0.15 |
Also: TESTS-step bug fixed (pack tests never ran since M1). Research briefs in ideas/*/RESEARCH.md, plastic-agent/TTT-DEEP-DIVE.md.
Campaign order (GPU lock): m6 → looped-latent v0 → stencil-focus v0 → ttt-context v0 → plastic-agent v0 (if GO).

## Resume after a CLI restart (2026-09-28)
Running detached (unaffected by the CLI): `arlab-campaigns` (runs ~/arlab-runs/_pilots/campaign-queue.txt one by one:
stencil-focus v0 running → ttt-context v0 → agentic-coding-small m7) and `arlab-p-pa-gate2` (plastic-agent gate pilot,
waits for the GPU lock; result in ~/arlab-runs/_pilots/pa-gate2.log: GO needs none ≤ 0.15 and icl − none ≥ 0.15;
on GO append "plastic-agent v0" to campaign-queue.txt, after the astra review is re-run on the redesigned pack).
On resume: `systemctl --user list-units 'arlab-*'`, `tail ~/arlab-runs/_pilots/campaigns.log`, read finished
reports (~/arlab-runs/<pack>/<tag>/report.md), update README Results/STATUS, re-arm a wake-up
(`until grep -q "end <pack> <tag>" ~/arlab-runs/_pilots/campaigns.log; do sleep 300; done` in the background).
Check m7's CALIBRATE timeout_rate first (m6 failed on load-dependent timeouts; TASK_SECONDS is now 5400).

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
