ACTIVE — build (M0–M6) is DONE; owner follow-ups in progress. NEW: tree search (Dream-RSI) — T0 built.

## Tree search (owner request 2026-09-28; plan ~/.claude/plans/effervescent-wandering-elephant.md; docs/TREE-SEARCH.md)
| Stage | State |
|---|---|
| T0 build | DONE (commits 2ac0ea9, f6a0cd4; astra review: 18 fixed, 4 recorded): arlab/tree/ (model, replay, policy sandbox + 3 built-ins, online TreeCampaign, dream, report, CLI); `make accept-T1` PASS (13 tests: no-leak replay, sandbox isolation, held-out dream guard, fixture tree run with kill -9 resume + pre-registered finalize → supported). README rewritten in plain language. 4 greedy campaigns imported to ~/arlab-runs/_tree/imported (smoke replay only). |
| T1 `_fixture` real-Codex shakedown | DONE: 3 paired rounds, 336 nodes; no dreamed-vs-fixed claim (noise ≥ effect); replay mis-ranked online; held-out guard rejected fx-d2. docs/TREE-SEARCH.md §Results |
| T2 nanochat-lite tree (root m3b:0005, seal matches) | QUEUED (relaunched 23:2xZ after a0 was underpowered; holdout seeds now 101–105): unit **arlab-tree-t2** (`_pilots/tree-t2.sh`, log `_pilots/tree-t2.log`) waits for arlab-campaigns to drain, then a0, a1 (controls) → dream n-d1 (held-out a1) → b1 (dreamed, if accepted) → g1 (greedy control); W4×32/round, ~20 GPU h |
| T3/T4 | **ttc-controller** built + astra-reviewed (static PASS, 39 tests; waits for GPU sampling: `build/sample.sh --pilot 100` then full ~3-5 GPU h). **latent-arch** built + 3 astra rounds (static PASS, 26 tests; next: full `arlab check` GPU PROBE + ~1-1.5 GPU-h pilot via build/pilot.sh). Both run after T2 in the GPU order. data-select, gpu-kernels later. |

Note: accept-M1 45/47 under GPU-campaign load (fixture wall-clock guard noise, see WORKLOG); re-run when the GPU queue is idle.

## Owner follow-ups (2026-09-28)
| Pack | State |
|---|---|
| agentic-coding-small | **m7 done: not_found_at_this_scale** (12 experiments, 0 timeouts, all discard; candidates 0.125–0.225 vs baseline seeds 0.15/0.25/0.225; upper bound 0.176 < MES 0.20) |
| looped-latent | **v0 done: not_found_at_this_scale** (upper bound 0.025 < MES 0.03, 20 loop designs) |
| stencil-focus | **v0 done: not_found_at_this_scale** (15 policies, best +0.017 < MES 0.05; 7 guard_fail on output_failure_rate; reminder ≈ off 0.108/0.109 vs pilot oracle +0.27) |
| ttt-context (TTT stage 0) | **v0 done: not_found_at_this_scale** (15 recipes; baseline TTT < no_ttt: holdout d = −0.024 ± 0.027; upper bound 0.003 < MES 0.05) |
| plastic-agent (TTT stage 1) | **v0 done: not_found (guard-limited)**: 15 experiments, 13 guard_fail, mostly battery_drop 0.03–0.07 > 0.02. Best 0001 hindsight_pairs success 0.696 (none 0.092, icl 0.579, twin 0.008, share 1.14) at battery_drop 0.04. The baseline itself dropped 0/.027/.007/.05/.03, so the pre-registered IDEA.md rule sets the guard to 0.0398. **v1 PAUSED: codex auth** since 23:49Z (refresh token invalidated; owner must re-login; auto-resumes) |

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
