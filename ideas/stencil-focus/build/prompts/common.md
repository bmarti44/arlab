## Context (read carefully)

We are building a synthetic benchmark of long mentor/mentee conversations. Somewhere inside long, unrelated
chat, the mentor states Python coding conventions (for example that function names must start with a given
prefix). Later a model must write code that follows the conventions still in force. Your text will be inserted,
sentence by sentence, into the mentor's turns of the conversation. It must read naturally in spoken English,
as a senior engineer talking to a junior colleague.

## Hard format rules (every string you write is validated by a script and rejected if it breaks one)

1. Each string is ONE sentence (templates render to one sentence). No line breaks.
2. No sentence-internal `.`, `!` or `?` followed by a space: this means no abbreviations such as "e.g.", "i.e.",
   "etc.", "vs.", "Mr.", "No.", no ellipses, no decimal numbers followed by a space, no URLs, no file names with a
   dot. Use commas, dashes (" - "), "so", "because", "and" instead. Complete sentences end with exactly one `.`,
   `!` or `?` at the very end; fragments (verb phrases) end with NO punctuation.
3. Plain ASCII only: straight quotes `'` and `"`, hyphen `-`. No emoji, no markdown, no backticks, no code
   blocks, no bullet points.
4. Placeholders are written exactly as given, in curly braces, e.g. `{affix}`; use each required placeholder
   exactly once and never invent new ones. Quote affix/module/decorator placeholders with single quotes exactly
   as shown in the examples, e.g. `'{affix}'`.
5. Do NOT copy the canonical instruction wording. Avoid any run of 5 or more consecutive words from the canonical
   texts listed at the end of this prompt (for example never write "always start function names with"). Vary the vocabulary and
   syntax a lot: different verbs, word orders, passive and active voice, conditions ("whenever you ..."),
   reasons ("so that ..."), hedges ("I'd really prefer ..."). Do not use the words "convention", "rule",
   "guideline" or "instruction" in more than 1 in 10 strings, so these words cannot be used to find them.
6. All strings in one list must be distinct and meaningfully different, not one template with a word swapped.

Write your result as ONE JSON file, exactly at the path given below, and nothing else in /out. Then print "done".
