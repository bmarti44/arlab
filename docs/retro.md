# arlab retro (M0–M6, 2026-09-26 → 2026-09-27)

## Outcome

| Campaign | Agent | Experiments | Active h | Rate | Verdict |
|---|---|---|---|---|---|
| nanochat-lite/m3b (M3) | gpt-6-sol | 28 (+2 interrupted by kill tests) | 2.90 | 9.6/h, median 5.2 min | **supported**: holdout d = 0.0151 val_bpb (d − 2·SE = 0.0102 > 0, all 3 seeds positive), MES 0.01 |
| digits-label-smoothing/m4 (M4) | gpt-6-sol | 20 | 0.41 | 48/h, median 1.1 min | not_found_at_this_scale (upper bound 0.026 < MES 0.05) |
| memory-longmemeval/m5 (M5) | gpt-6-sol | 25 | 1.99 | 12.5/h, median 2.0 min | not_found_at_this_scale (upper bound 0.062 < MES 0.08), stop no_keep |
| agentic-coding-small/m5c (M5) | gpt-6-sol | 3 | 2.92 (incl. 3 calibration passes) | 1.0/h, median 70 min ("speed target missed") | inconclusive: stopped_early:max_hours (2 guard_fail on timeouts, 1 discard; 0002 scored 0.20 vs 0.15 but 25%+ timeouts) |

Scripted/fixture campaigns (M1, M2) reached every status and every verdict; the M2 anti-cheat cases (b, c, e, f, g)
behaved as specified.

## Where GPU time went
- nanochat-lite: RUN dominates (2.33 of 2.90 active h in m3b); EVALUATE 0.12 h; the agent only 16% of a cycle, so
  the M3 fallback (a speculative proposal in flight) was not needed.
- memory-longmemeval: the baseline pass takes 26 s, but candidates that spend the per-question budget (an LLM pass
  over every session) run 20+ minutes; RUN 1.38 h vs propose 0.61 h.
- agentic-coding-small: a 40-task validation pass costs ~45–60 min with Qwen3.5-4B (~160–250 generated tok/s in
  aggregate). That, not the agent, bounds the campaign to a handful of experiments.
- Pilots/calibration outside campaigns: nanochat pilot (5 × 4 min), memory model selection (Qwen3-0.6B 0.107,
  Qwen3-1.7B 0.291), agentic difficulty ladder (30 tasks), prefix-cache A/B, two nanochat re-probes.

## Accept rates
- nanochat m3b: 1 keep / 28 (the optimizer batch 2^17 → 2^16 tokens; the same change the scripted M2 test used as
  its "known-good" edit, found independently by the agent in experiment 0005).
- digits m4: 1 keep / 20 on validation (0.964 → 0.984) that did not hold on the holdout.
- memory m5: 0 / 25 (best: an LLM-rewritten extra query, +0.005; per-session fact extraction collapsed to 0.09).
- agentic m5c: 0 / 3 — two candidates failed the timeout guard (the baseline itself exceeds it), one discard.

## Agent latency and tokens (gpt-6-sol, effort high)
| Campaign | Calls | Input tokens (cached) | Output tokens | Median propose |
|---|---|---|---|---|
| nanochat m3b | 33 | 3.84M (3.16M) | 61k | 53 s |
| digits m4 | 21 | 2.09M (1.76M) | 47k | 62 s |
| memory m5 | 28 | 2.84M (2.31M) | 81k | 84 s |
| agentic m5c | 3 | 0.27M (0.20M) | 5k | 47 s |
gpt-6-astra: pack reviews (digits, memory, agentic) and ~100 generated coding tasks (10 pilot, 20 ladder, 70 main).
gpt-6-sol (xhigh): four code-review rounds of arlab itself.

## What broke (and what was changed)
- **Codex blocked by Claude Code's auto-mode classifier** (B1) until the owner approved the containerized call.
- **Fixture timing flakiness** (M1): big.LITTLE core placement and CPU contention made wall-clock guards flaky →
  RUN/EVALUATE pinned to the fast cores (cpuset), bigger fixture batch, guard 1.5×.
- **NVIDIA hook needs a writable /run** → EVALUATE mounts the RUN output at /run_out.
- **nanochat m3 calibration anomaly**: sigma 0.0185 vs 0.0027 (two seeds +0.03–0.045 val_bpb after a startup
  recompile); not reproducible afterwards (quiet and CPU-hog re-probes normal). The power check refused to search
  (inconclusive: underpowered); re-run m3b calibrated normally. Root cause unconfirmed (torch.compile/autotune state).
- **Code review (gpt-6-sol xhigh, 4 rounds, ~25 findings)**: path-unsafe tags, name-based `docker rm`, cidfiles the
  agent could plant (resume could have killed a foreign container), traversal/symlinks into the evaluator, Docker
  launch failures counted as candidate crashes, lost agent-call usage after kill -9, per-item token caps checked
  only before a call, weak accept checks, a reversed report label, clean-room report forgery (defense in depth
  only — inherent to in-process pytest), kill-script and symlink-loop edge cases. All fixed or recorded.
- **Pack reviews (gpt-6-astra)**: memory — gold reachable via a symlinked answers file, `has_answer`/`answer_…`
  session ids/`_abs` question ids leaking labels, number normalization collisions (20.5 = 25), fail-open scoring;
  agentic — fail-open scoring, timeouts masked by budget status, orphaned background processes; digits — labels
  recoverable from sklearn's bundled dataset. Fixed; in-process isolation limits recorded.
- **agentic-coding-small sizing**: astra's first tasks were beyond the 4B model (0/10); a difficulty ladder (L1 4/5,
  L2 3/5, L3 3/5, L4 1/5) set the target at L3–L4. Then two infrastructure lessons: at 16-way the 1,200 s wall
  clock was hit by 55% of tasks (guard unreachable, results load-dependent, sigma 0.07 between runs); at 40-way,
  2,048-token replies outlasted the client's 300 s timeout and retries snowballed. m5c: 20-way, long client
  timeout, step/token budgets with the wall clock as a safety net, and a seed pack (runs are not repeatable).
- Prefix caching is off by default for hybrid (Mamba/GDN) Qwen3.5 in vLLM 26.04; enabling it cut a repeated 24k-token
  prefill from 4.1 s to 0.2 s.

## Would do differently
- Pilot service packs at the final concurrency with per-step timing before generating tasks; measure run-to-run
  determinism before choosing deterministic-pack seeds.
- For agentic-coding-small specifically: the next tag should size the wall-clock limit (or the timeout guard) from
  m5c's measured distribution, or cut generated tokens per task (edit-based tools), so that a 6-hour campaign has
  room for more than three candidates.
- Keep the box quiet during any calibration, and give each GPU calibration its own inductor cache or disable
  autotuning, to rule out compile-state effects.
