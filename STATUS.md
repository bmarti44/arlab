ACTIVE — build (M0–M6) is DONE; owner follow-ups in progress. NEW: tree search (Dream-RSI) — T0 built.

## Tree search (owner request 2026-09-28; plan ~/.claude/plans/effervescent-wandering-elephant.md; docs/TREE-SEARCH.md)
| Stage | State |
|---|---|
| T0 build | DONE: arlab/tree/ (model, replay, policy sandbox + 3 built-ins, online TreeCampaign, dream, report, CLI); `make accept-T1` PASS (13 tests: no-leak replay, sandbox isolation, held-out dream guard, fixture tree run with kill -9 resume + pre-registered finalize → supported). README rewritten in plain language. 4 greedy campaigns imported to ~/arlab-runs/_tree/imported (smoke replay only). |
| T1 `_fixture` real-Codex shakedown | NEXT: `~/arlab-runs/_pilots/tree-t1.sh` (r0 π₀ W8×48 → dream fx-d1 → r1 dreamed + c1 control → … 3 rounds, ~450 Codex calls), unit arlab-tree-t1, CPU only |
| T2 nanochat-lite tree (root m3b keep) | after the GPU campaign queue |
| T3/T4 | second chances; new packs ttc-controller, latent-arch, data-select, gpu-kernels |

## Owner follow-ups (2026-09-28)
| Pack | State |
|---|---|
| agentic-coding-small | m6 stopped (load-dependent timeouts); fix TASK_SECONDS 5400; **m7 queued** |
| looped-latent | **v0 done: not_found_at_this_scale** (upper bound 0.025 < MES 0.03, 20 loop designs) |
| stencil-focus | **v0 running** (campaign queue) |
| ttt-context (TTT stage 0) | v0 queued after stencil |
| plastic-agent (TTT stage 1) | redesigned; gate-2 pilot waiting for the GPU lock (arlab-p-pa-gate2) |

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
| accept-T1 | PASS (tree search, 13 tests) |

BLOCKED.md: B1 (containerized Codex) resolved by the owner on 2026-09-26.
Docs: README.md (results), docs/retro.md, docs/ORCHESTRATOR.md, docs/PACK-AUTHORING.md, DECISIONS.md, WORKLOG.md.
