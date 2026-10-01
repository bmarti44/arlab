# BLOCKED

## B1 — Containerized Codex call denied by Claude Code's auto-mode classifier (2026-09-26)
- **Check:** M0 "three containerized `codex exec` calls return schema-valid JSON" (and everything using Codex: M3, M4 astra review/agent experiments, M5).
- **Evidence:** the exact PLAN §3.6 `docker run … arlab-agent:0.157.1 codex exec … --dangerously-bypass-approvals-and-sandbox …` was refused by the auto-mode classifier ("Create Unsafe Agents"). The `.claude/settings.json` allow rule `Bash(docker:*)` did not override it.
- **Attempts:** none. The denial says not to work around it, so none were made; launching the runner with the Codex backend would be the same outcome and is also on hold.
- **Owner decision needed:** allow Claude Code to launch the containerized Codex agent (and the arlab runner that launches it). For example, approve it interactively, or add a permission rule you're comfortable with. Everything that doesn't need Codex (M0 GPU/vLLM parts, M1, M2) continues meanwhile.
- **Resolved 2026-09-26 23:35Z:** the owner approved running Codex without the sandbox. A test `codex exec` call (gpt-6-sol) returned a schema-valid proposal and made the edit.

## B2: latent-arch (T4.2) program task not learnable at pack scale (2026-10-01)
- **Check:** the IDEA.md GPU pilot gate requires the baseline to score 30–70 % in-distribution (ID) on the synthetic programs before calibration.
- **Evidence:** text is learned normally in every arm (val_bpb 1.15–1.25). Program accuracy is at chance in every arm, at every depth including k = 1.
  - Pilot (mod 100, programs 9 % of rows, 330 s): baseline 1.8 % ID, depth12 1.2 %, loop_naive 0.5 %, depth12_2x 1.8 %. Floor (answer = last constant) 1.7 %.
  - Sweep 1: mod 10 at 48/16 and 58/6 rows → 9.4 % / 8.4 % (chance 10 %); mod 100 at 32/32 → 1.7 %.
  - Sweep 2: chains only (p_bin 0), mod 10, 32/32 → 9.9 %; 4/60 (94 % programs) → 9.1 %. A direct probe of that checkpoint on its own *training* programs gave 8.6 % with only 2 distinct predictions (the modal digit). The data is correct (decoded rows; the train stream includes the answer) and so is the evaluator.
  - Attempt 3: the same 94 %-program config for 1800 s (about 7,300 steps) → 10.4 % ID, k1 9.3 %. Train loss plateaued at 1.40–1.49.
- **Attempts:** 3 (knob sweep, learnability sweep, 5.5× longer training). All are recorded in WORKLOG.md. The pack is restored to its committed state (mod 100, 58/6). Pilot outputs are in ~/arlab-runs/_pilots/latent*/.
- **Reading:** a 26M from-scratch GPT doesn't get past the variable-binding plateau of this task within 1–7k steps. Multi-hop binding tasks are known to plateau and then jump suddenly, and here that jump needs far more steps than a 5–30 min budget gives.
- **Options (owner):**
  1. A simpler serial task that small models learn quickly, such as pointer chasing without arithmetic, or permutation composition / state tracking (S5-style).
  2. Warm-start from a checkpoint pretrained on the task family, then search architectures for depth extrapolation. This is no longer "from scratch".
  3. Hours-long runs per experiment, for about 20–40 GPU h per campaign.

  Status: partial. Moving on to the next GPU item.
