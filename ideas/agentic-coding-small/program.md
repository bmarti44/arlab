# agentic-coding-small — instructions for the research agent
You make ONE change per call. The runner (not you) runs the evaluation, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator, the tasks or the data; do not try.
## Goal
maximize pass_rate: the fraction of multi-step Python coding tasks a small model (Qwen3.5-4B, 32k context,
temperature 0) solves using YOUR agent scaffold. Runs are not exactly repeatable (server batching), so a change is kept
only if it beats the incumbent on one run and then clearly (by more than 2·SE) on a fresh run; the holdout decides.
## What you may change
agent.py only: solve(task, llm, tools). Nothing else has any effect. The only model access is
llm.chat(messages, max_tokens=None, stop=None) -> str (completions ≤ 2,048 tokens; llm.tokens_left = what is left of
this task's 400,000-token budget; going over ends the task). Tools (see frozen_run/codetools.py): read, write, edit,
run(cmd, timeout=60) and finish(); ≤ 30 tool calls per task (plus a 2,400 s safety limit). agent.py must not read files, the
environment or the network except through these tools.
## What the code does
The harness (frozen_run/harness.py) runs 20 tasks concurrently. Each task starts in a fresh working directory
with its starter files; task = {id, title, instructions, files}. When solve() returns, hidden pytest tests run on
the files the instructions name. Every expected test must pass for the task to count.
The baseline is a bash-only loop: THOUGHT + one ```bash block per turn, last 12 turns of history kept.
## Ideas worth trying (from IDEA.md)
Planning and task tracking (a short plan, a checklist of required functions), self-verification (write and run
quick tests of what the instructions specify before finishing), context management (summarize old turns, keep file
contents fresh), better tool use (direct write/edit instead of heredocs, reading files first), recovery from
errors and format mistakes, stopping rules (don't finish before tests pass), prompt wording.
## Rules
- One change, one hypothesis_tag (reuse an existing tag for the same idea).
- Read history.md: don't repeat a failed idea unless you change it materially. After 3 discards in a row, try something structurally different.
- Respect constraints.md (facts about this machine).
- Prefer simple changes; equal results with less code are better.
- Keep notes.md short and useful to your future self.
- Finish with the JSON object required by the output schema and nothing after it.
