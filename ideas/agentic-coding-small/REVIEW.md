Found the following issues. No files were modified.

**Evaluator gaming**

- [harness.py:66](/pack/frozen/run/harness.py:66): Surface code runs inside the trusted harness process and can alter `__main__.client`, limits, timing, and budget-reporting methods without editing frozen files. **Fix:** Run the surface in an isolated subprocess; keep accounting and result generation in the parent.
- [codetools.py:79](/pack/frozen/run/codetools.py:79): `run()` uses `cwd` as its only filesystem boundary; shell commands can access sibling task workspaces and modify another task’s submission after it finishes. **Fix:** Isolate each task’s filesystem and seal its output when execution ends.

**Leakage**

- **none found in the supplied data flow.** Reconstructed splits contain 40 disjoint tasks each; no exact duplicate instructions, starters, tests, or solutions were found. PREPARE exports only public task fields. Actual mount isolation depends on infrastructure outside this snapshot.

**Budget loopholes**

- [harness.py:35](/pack/frozen/run/harness.py:35), [harness.py:89](/pack/frozen/run/harness.py:89): Surface code or shell commands can call `http://llm:8000` directly, bypassing the scoped client’s token/completion limits and its exported accounting. **Fix:** Require an authenticated metering proxy and block direct service access.
- [codetools.py:28](/pack/frozen/run/codetools.py:28), [harness.py:43](/pack/frozen/run/harness.py:43): The surface receives mutable step counters, limits, deadlines, and the underlying client; resetting `tools.steps` permits additional calls. **Fix:** Own these controls outside the surface process and expose only validated RPC operations.
- [harness.py:46](/pack/frozen/run/harness.py:46), [harness.py:84](/pack/frozen/run/harness.py:84), [codetools.py:85](/pack/frozen/run/codetools.py:85): Deadlines are checked cooperatively; running model calls and arbitrary surface computation are not interrupted, and detached shell workers can survive task completion. **Fix:** Enforce deadlines externally and terminate the entire task cgroup before collecting output.

**Scorer bugs**

- [evaluate.py:36](/pack/frozen/eval/evaluate.py:36): Only `budget_exceeded` disqualifies a submission; missing statuses, crashes, `step_limit`, and `time_limit` can all receive full credit. An empty status file also yields zero reported timeouts. **Fix:** Require exactly one valid terminal record per expected task and reject limit/error outcomes before scoring.
- [harness.py:77](/pack/frozen/run/harness.py:77), [evaluate.py:45](/pack/frozen/eval/evaluate.py:45): `budget_exceeded` overwrites `time_limit`, causing tasks exceeding both limits to disappear from the timeout guard. **Fix:** Record timeout and budget violations independently.

Read-only probes confirmed shell escape, counter reset, and fail-open status scoring. The external `arlab` client and clean-room implementation were unavailable, so their additional protections—and resistance to pytest/JUnit tampering—remain unverified.
---
## Orchestrator response (2026-09-27)
- Scorer fail-open: fixed — a task scores only with exactly one terminal record whose status is finished /
  returned / step_limit / time_limit; missing, crashed or over-budget tasks score 0.
- `budget_exceeded` hiding timeouts: fixed — the harness records `timed_out` independently; the guard uses it
  (a missing record counts as timed out).
- Detached shell workers: fixed — the harness kills every process group its tools started when a task ends.
- Sibling workspaces: a task's commands can touch other tasks' directories; that can only damage other tasks
  (hidden tests still decide each task), not pass them unsolved. Recorded, not changed.
- In-process surface (mutable counters/limits/client) and direct `http://llm:8000` access: by design (PLAN §6.3
  harness calls the surface in-process); the runner's absolute service_tokens budget is measured from vLLM and
  cannot be bypassed; per-task caps are advisory against deliberate bypass (DECISIONS.md 2026-09-26, review r3).
- Cooperative deadlines: model calls and tool calls both check the deadline (TimedClient); a surface busy-looping
  without either is stopped by RUN's own timeout (a crash).
