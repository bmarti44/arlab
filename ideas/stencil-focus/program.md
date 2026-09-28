# stencil-focus — instructions for the research agent
You make ONE change per call. The runner (not you) runs the evaluation, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator or the data; do not try.

## Goal
maximize fraction_required: how well a frozen Qwen3-4B follows the coding conventions its mentor stated in a long
mentor/mentee conversation (12-30 sessions, ~8-25k tokens) when it only sees a 3,584-token window. The newest part
of the conversation fits the window; everything older is cut. Your policy decides which earlier sentences are
restated in a short reminder inside the window. Noise: 300 validation items, paired per item (SE ~ 0.01-0.02). A
change is kept only if it beats the incumbent on one seed and then clearly (by more than 2·SE) on a fresh seed.

## What you may change
surface/stencil.py only: plan(sentences, current, query, window) -> Plan (read frozen/run/window.py, it is short).
- sentences: every sentence of the earlier sessions (0..s-1), chronological, with id, session, line, speaker
  (mentor / mentee / other), text, tokens and evicted_at_base_window (cut away when there is no reminder).
- current: the current session (always in the window). query: what the model is asked to write,
  e.g. "function that implements merge sort" or "Heap class with insert and heapify methods".
- window: tokens(text), render(ids, header), reminder_tokens(ids, header), pack_newest_first(ids, budget, header),
  displaced(reminder_tokens) = ids cut away when the reminder costs that many tokens, build(plan) = exact prompt.
- Plan(ids, header, placement, budget): ids = ordered sentence ids (each once); header = index into
  window.HEADERS (1 = none); placement = "after_thread" | "before_thread"; budget <= 1024 tokens, and the rendered
  reminder must fit it. The reminder takes window space: a bigger reminder cuts more of the thread.
You select, order, size and place whole verbatim sentences. You never write reminder text yourself.
Pure Python (stdlib + `from window import ...`): no files, argv, environment, network, no model or embedder
loads, no LLM calls. plan() is called once per item; keep it fast (a few ms per item; train_s is guarded at 1.3x).

## What the code does
The harness builds each item's window, calls plan(), renders the reminder ("- sentence" per line under the
header), truncates the thread from the oldest side so the prompt fits 3,584 tokens, and asks the model once
(greedy, no thinking) to write the code for the query. The baseline is stencil-llm's shipping policy: all mentor
sentences evicted at the base window, newest first, into 256 tokens, header 0, after the thread.

## Scoring (be exact)
A frozen regex/AST checker scores the generated code against the conventions still in force at the last session
(naming prefixes/suffixes/case/characters, type annotations, try/assert, docstrings, comments, imports,
decorators). fraction_required = mean over the checks whose family the query requires: class/method families when
the query names a class or method, function families otherwise, variable/import/comment always; a missing
required class or function scores 0. Using the query to prefer the conventions it will be checked on is
legitimate (it exploits this frozen rule, and that is allowed). Guard: output_failure_rate (no code, cap hit,
repetitive or timed-out outputs) must stay <= 1.15x the baseline's.

## Hard rules (the run is `invalid` otherwise)
- Only whole verbatim sentences via ids; every recorded prompt is rebuilt and compared by the evaluator.
- surface/stencil.py must not contain any 5-word run from MemoryCode's topics.json instruction texts or eval
  queries (no lists of known conventions or tasks). Write general logic, not memorized instruction strings.
- plan() must not raise and must return a valid Plan for every item.

## Ideas worth trying
- Separate convention-like mentor sentences from chat about other things (the conversation is long and mostly
  unrelated; the mentor also gives advice and workplace instructions that are not code conventions).
- Supersession: conventions get updated; the newest statement for the same thing should win, older ones dropped.
- Size: spend more or fewer tokens (up to 1,024) — but check whether the gain is selection or just size.
- Query conditioning: prefer conventions about the object the query asks for (class/method vs function).
- Order and placement of the reminder, header choice, deduplication, restating in-window statements or not
  (displaced(B) tells what the reminder itself pushes out of the window).

## Rules
- One change, one hypothesis_tag (reuse an existing tag for the same idea).
- Read history.md: don't repeat a failed idea unless you change it materially. After 3 discards in a row, try something structurally different.
- Respect constraints.md (facts about this machine).
- Prefer simple changes; equal results with less code are better.
- Keep notes.md short and useful to your future self.
- Finish with the JSON object required by the output schema and nothing after it.
