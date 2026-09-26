# memory-longmemeval

**Hypothesis.** A better memory subsystem (write / compress / retrieve / assemble) around a fixed small model
improves long-conversation QA accuracy under a fixed per-question token budget.

**Data.** LongMemEval `_s` (xiaowu0162/longmemeval-cleaned, revision 98d7416c; 500 questions, ~48 sessions /
~115k tokens of history each). PREPARE drops `single-session-preference` (rubric-style gold, 30 questions) and
every question whose gold has an alias longer than 5 normalized words (55 more), keeping 415 short-span questions.
They are split with a fixed seed, stratified by type (abstention items are their own stratum), into equal halves:
**207 validation / 207 holdout**. Aliases come only from the dataset's own answer text
("X. Y (including the last day) is also acceptable.", "X (or Y)").

**Model.** Served by vLLM 26.04 from the HF cache, temperature 0, no thinking: the smallest cached Qwen3/Qwen3.5
model whose baseline accuracy lands in [0.2, 0.8] (chosen at check time; recorded in DECISIONS.md).

**Budget (fixed before calibration).** An absolute cap of **160,000 service tokens per question** (prompt +
completion; ≈ one short pass over every session of the ~115k-token haystack plus answering), i.e.
`service_tokens ≤ 160,000 × 207 = 33,120,000` per split. Answers are cut to 32 words.

**Scoring.** Deterministic: an item scores 1 iff the normalized answer (lowercase, punctuation and articles
removed, number words → digits) exactly equals a normalized gold alias; abstention questions need an answer from
the frozen abstention phrase set ("I don't know", …). Hedged or multi-candidate answers score 0. No LLM judge.

**MES (fixed before calibration).** mes ≥ 2.5·√(0.2/207) = 0.0777 → **0.08** (8 points of accuracy).
Seeds: deterministic-pack defaults (temperature 0): calibration [1, 2], screen 1, confirm [], holdout [101].
