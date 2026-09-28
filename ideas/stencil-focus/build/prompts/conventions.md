# Task: paraphrase templates for coding conventions (verb phrases)

__COMMON__

## What to write

For each convention KIND below, write exactly 9 distinct **verb-phrase templates**. A verb-phrase template is a
bare infinitive verb phrase, starting with a lowercase verb, with no final punctuation, that completes all of
these frames grammatically:
"I'd like you to <VP>." / "Please <VP>." / "Remember to <VP>." / "<VP>, it keeps reviews quick."
(e.g. "prefix every {obj} name you write with '{affix}'", "make sure each {obj} name opens with '{affix}'").

The meaning must be exactly the canonical meaning (a script checks the generated code with a regex; a paraphrase
that means something else breaks the benchmark). Make ~3 of the 9 per kind longer and more conversational
(10-22 words, e.g. with a "whenever you ..." clause or a short reason), the rest 5-12 words. Only use the
placeholders listed for the kind; everything else is literal text.

Placeholders and the values they will take:
- `{obj}` singular noun phrase: one of "function", "variable", "method", "attribute", "function argument", "class"
  (used as a modifier, as in "{obj} names"). `{objs}` plural: "functions", "variables", "methods", "attributes",
  "function arguments", "classes".
- `{affix}` a short string such as a_  x_  fn_  _b  _vr  (written '{affix}' with single quotes).
- `{module}` a Python standard-library module name such as secrets, gzip (written '{module}').
- `{decorator}` a decorator name with its at-sign such as @retry, @timer_class (written '{decorator}').
- `{case}` one of "all uppercase letters", "CamelCase", "snake_case" (written without quotes).

KINDS (key: placeholders; canonical meaning; canonical wording to avoid copying):
- `case`: {case}; class names must be written in {case}; "always use CamelCase for class names".
- `chx`: {obj}; every {obj} name must contain the literal string 'chx' (write 'chx' literally, with quotes);
  "always include the string 'chx' in function names".
- `prefix`: {obj}, {affix}; every {obj} name must START with '{affix}'; "always start function names with 'a_'".
- `suffix`: {obj}, {affix}; every {obj} name must END with '{affix}'; "always end method names with '_x'".
- `annotation`: {objs}; always add type annotations (argument and return type hints) to {objs} (functions or
  methods); "always use annotations for functions".
- `try`: {objs}; every one of the {objs} must contain a try statement (try/except block);
  "always include try statements in methods".
- `assert`: {objs}; every one of the {objs} must contain assert statements; "always include assert statements in functions".
- `docstring`: {objs}; every one of the {objs} must have a docstring; "always use docstrings in methods".
- `comment`: no placeholder; always add comments to the code; "always add comments in your code".
- `import`: {module}; always import the '{module}' module, even when it is not used;
  "always import the 'secrets' module even if it is not used".
- `decorator`: {decorator}, {objs}; always put the '{decorator}' decorator from the 'pedantic' package on all
  {objs} (write 'pedantic' literally); "always add the '@retry' decorator from the 'pedantic' module to all functions".
- `digit`: {obj}; every {obj} name must include a single digit (a digit at the end of the name is what is meant,
  e.g. a name like total3; you may say "end with a single digit" or "include a single digit");
  "always include a single digit in function names".

## Output

Write `/out/conventions.json`:
```
{"case": [9 strings], "chx": [...], "prefix": [...], "suffix": [...], "annotation": [...], "try": [...],
 "assert": [...], "docstring": [...], "comment": [...], "import": [...], "decorator": [...], "digit": [...]}
```
