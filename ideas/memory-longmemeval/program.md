# memory-longmemeval — instructions for the research agent
You make ONE change per call. The runner (not you) runs the evaluation, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator or the data; do not try.
## Goal
maximize accuracy on long-conversation memory questions (~48 past chat sessions, ~115k tokens, per question).
The pack is deterministic (temperature 0), so a change is kept if it beats the incumbent by more than 2·SE
on the validation questions; the one-shot holdout decides at the end.
## What you may change
memory.py only: class Memory(llm, tools) with ingest(session) and answer(question, question_date) -> str.
Nothing else has any effect. The only model access is llm.chat(messages, max_tokens) (temperature 0, answers
and calls capped); llm.tokens_left tells you what is left of this question's 160,000-token budget; going over
makes the answer empty. tools.BM25(docs).search(query, k) and tools.tokenize(text) are the frozen helpers.
memory.py must not read files, the environment or the network.
## What the code does
The harness creates one Memory per question, calls ingest() for every session in chronological order
(session = {session_id, date, turns: [{role, content}]}), then answer(). Up to 16 questions run concurrently.
The baseline stores turns, retrieves the top-12 BM25 turns and asks for a short answer.
## Scoring (be exact)
The answer is normalized (lowercase, no punctuation/articles, number words → digits) and must EXACTLY equal the
gold: a short phrase, number or name (≤ 5 words), e.g. "Business Administration", "7 days", "4", "Target".
If the history does not contain the answer, reply exactly "I don't know". Hedges, lists of candidates,
explanations or extra words score 0. Answers are cut to 32 words.
## Ideas worth trying (from IDEA.md)
Better retrieval (query rewriting, session-level retrieval, dates, neighbour turns), writing compressed notes or
facts per session during ingest (one short LLM pass per session fits the budget), knowledge-update handling
(latest fact wins), temporal arithmetic with the session dates, answer-format control, abstention calibration.
## Rules
- One change, one hypothesis_tag (reuse an existing tag for the same idea).
- Read history.md: don't repeat a failed idea unless you change it materially. After 3 discards in a row, try something structurally different.
- Respect constraints.md (facts about this machine).
- Prefer simple changes; equal results with less code are better.
- Keep notes.md short and useful to your future self.
- Finish with the JSON object required by the output schema and nothing after it.
