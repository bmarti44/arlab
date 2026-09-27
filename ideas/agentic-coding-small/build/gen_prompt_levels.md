# Generate __N__ coding tasks for evaluating a small (4B) coding agent

Write exactly __N__ self-contained Python coding tasks into /out/tasks/, one directory per task, ids
`__PREFIX__-01` … `__PREFIX__-__N__`. Themes for this batch (one task each, in order): __THEMES__.

A small local model (Qwen 4B class, 32k context) will solve each task by itself inside a fresh working directory
containing only the starter files, using shell commands (python3, pytest, cat, sed …), in ≤ 30 steps. It sees
the instructions and the starter files, never the hidden tests. Hidden tests then run in a clean room with only
the files listed in `sources` plus the tests. __LEVELS__

## Each task directory `/out/tasks/<id>/`
- `task.json`: `{"id": "<id>", "title": "...", "difficulty": "<level name given below>",
  "instructions": "<markdown>", "sources": ["relative/paths.py", ...], "expected": ["test_<id_underscored>.py::test_name", ...]}`
- `starter/`: the files the agent starts with (at least one file). Starter code must NOT pass the hidden tests
  (stubs raising NotImplementedError, missing functions, or a realistic bug to fix).
- `tests/test_<id_underscored>.py`: ONE flat hidden pytest file (no conftest, no fixtures from files, no classes
  needed), 5–10 test functions; each id in `expected` is `<file>::<function>`, listing every test function exactly.
- `solution/`: a complete reference solution: every path in `sources`, final content.

## Requirements (all must hold; the tasks are validated automatically and rejected otherwise)
1. Size and difficulty exactly as the level definitions above say (this batch calibrates what the model can do).
2. `instructions` fully specify everything the tests check: module paths, class/function names, signatures, return
   types, exact error types, edge cases, output formats. Tests must not check anything the instructions don't state.
   Include a short "Files" section naming each file in `sources`.
3. `sources` lists every file the tests import (including package `__init__.py` files) and nothing else; test files
   import them from the directory root (e.g. `from inventory.store import Store`). Tests never import starter-only files.
4. Python 3.12 standard library only (tests: pytest only). No network, no subprocess, no randomness without a fixed
   seed, no wall-clock timing, no reading files outside the working directory (use pytest's tmp_path for files).
5. Tests pass on `solution/` and each test fails on `starter/`; the whole test file runs in < 5 s.
6. Tasks are realistic small-software work (parsers, data structures, CLIs as functions, text processing, small
   simulations, state machines, refactors with behaviour tests, bug fixes in existing modules) — not puzzles or
   trivia. Every task distinct.

Write only under /out/tasks/. When done, print one line per task: `<id> <difficulty> <title>`.
