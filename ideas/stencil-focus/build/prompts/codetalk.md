# Task: mentor talk about code that is NOT a checkable coding convention

__COMMON__

## What to write

Distractor sentences a senior engineer says about code and engineering work. They must sound as "codey" as the
real conventions (mention functions, methods, classes, variables, names, modules, imports, tests, reviews,
refactors, type checkers, exceptions) but must NOT ask for any of the following checkable conventions, and must
not contradict them either:
naming prefixes/suffixes or any rule about how names are spelled (case, digits, required substrings); type
annotations or type hints; try/except blocks; assert statements; docstrings; comments; importing specific modules;
adding decorators. Talking ABOUT these things in a non-directive way is fine and wanted, e.g. "The linter flagged
three functions in the billing module yesterday." or "I renamed the parser class last sprint, so the old imports
broke." Never tell the listener to follow a naming or style rule.

1. `advice`: 90 distinct **verb-phrase** templates (bare infinitive, lowercase verb first, no final punctuation,
   5-18 words) giving code-related advice that is not one of the conventions above, e.g.
   "run the whole test suite before you push a refactor", "keep each function focused on one job",
   "read the stack trace from the bottom up when a method blows up". They must complete
   "I'd like you to <VP>." / "Please <VP>." grammatically.
2. `talk`: 150 distinct complete sentences (8-28 words, ending with `.`, `!` or `?`) of mentor talk about code:
   anecdotes, observations, questions, opinions, status updates. About a third should mention names, functions,
   methods, classes, variables, modules or imports in a descriptive (non-directive) way.

## Output

Write `/out/codetalk.json`: `{"advice": [90 strings], "talk": [150 strings]}`.
