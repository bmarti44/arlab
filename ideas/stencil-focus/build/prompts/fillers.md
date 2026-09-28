# Task: paraphrases of workplace instructions that are NOT about code style (verb phrases)

__COMMON__

## What to write

Below are workplace instructions a mentor gives (tools, habits, office life). They are distractors in the
benchmark: they look like instructions but are not coding conventions. For EVERY variant text below write exactly
3 distinct **verb-phrase paraphrases**: bare infinitive verb phrases starting with a lowercase verb, no final
punctuation, completing "I'd like you to <VP>." / "Please <VP>." / "<VP>, it keeps things simple." grammatically,
5-18 words, same meaning (keep the named products and numbers exactly: Vim, Zoom, 50 minutes, ...). Negative
variants ("never ...") must stay clearly negative (e.g. "stay away from ...", "don't ...", "avoid ..."). Vary the
wording between the 3 paraphrases. There are no placeholders in this task.

Variants (id: [variant 0, variant 1, ...]):
__FILLERS__

## Output

Write `/out/fillers.json`: `{"<id>": [[3 strings for variant 0], [3 strings for variant 1], ...], ...}` with the
same ids and the same number of variants, in the same order, as above.
