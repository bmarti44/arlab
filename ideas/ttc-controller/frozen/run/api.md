# ttc-controller — the controller API (surface/controller.py)

The harness replays a fixed cache of 32 sampled GSM8K reasoning traces per problem (Qwen3-1.7B, non-thinking).
Your file defines one class; the harness drives it inside a sandboxed process that it starts fresh for every
replicate and budget level.

```python
from ttc_api import BudgetExhausted       # optional import; the only harness module you may use

class Controller:
    def __init__(self, cfg: dict): ...     # cfg: budget, n_problems, pool, max_traces, seed
    def fit(self, train: list) -> None: ...        # optional, <= 60 s together with __init__
    def solve(self, problem) -> str | None: ...    # <= 5 s per call; return a label or None (abstain, scores 0)
```

**cfg.** `budget` = B at this level (mean generated tokens per problem; the harness also plays 0.5·B and 2·B),
`n_problems`, `pool` = budget × n_problems (the whole allowance of the episode), `max_traces` = 32, `seed` (for your
own RNG; differs per replicate).

**Problems** arrive one at a time in a seeded order. `problem` has:
- `problem.pool_left` (tokens left in the pool), `problem.problems_left` (including this one), `problem.index`,
  `problem.budget`, `problem.max_traces`, `problem.spent` (tokens spent on this problem), `problem.traces`
  (the traces opened so far, in order);
- `problem.open()` → the next trace in this problem's fixed random order, or `None` after `max_traces`.
  Opening is free; you cannot choose or skip traces.

**Traces.** `trace.read(n=None)` reveals up to `n` more tokens (rounded UP to a multiple of 32; `None` = to the
end) and returns how many were revealed. Exactly the revealed tokens are charged to the pool. A read is cut short
at the end of the trace or when the pool runs out; with the pool empty it raises `BudgetExhausted` (if it escapes
`solve()`, the problem is abstained). Revealed per-token signals (float32 numpy arrays of length `trace.pos`):
- `trace.logprob` — log-probability of the sampled token;
- `trace.conf` — DeepConf token confidence, −mean of the top-5 log-probabilities (higher = more confident);
- `trace.ent` — entropy of the renormalized top-5 distribution.

Once a trace has been read to its end (`trace.done`), `trace.length`, `trace.finish` (`"stop"` or `"length"` =
truncated at 1024 tokens) and `trace.answer` are set. `trace.answer` is an **opaque label** (`"a0"`, `"a1"`, … in
the order distinct answers were first revealed in this problem; `None` if the trace was truncated or has no boxed
answer). You never see answer values, the question, the problem id, or unrevealed tokens.

**Scoring.** `solve()` must return a label produced by a trace read to its end in this problem (anything else makes
the run invalid) or `None`. A problem scores 1 if the label's answer equals the gold answer. The metric is accuracy
at B averaged over 8 replicates (different problem orders and trace orders); the guards need accuracy at 0.5·B
and 2·B to stay ≥ 0.97× the baseline's, and the whole RUN (24 episodes) to take ≤ 600 s.

**fit(train).** `train` is a list of labelled train problems (disjoint from every scored problem). Each is a list
of 32 dicts: `logprob`, `conf`, `ent` (full float32 arrays), `length`, `finish`, `label` (opaque as above, in the
stored order) and `correct` (bool: this trace's answer is right). Use it to learn stopping rules, vote weights or
filters. It is the same data in every episode; nothing you compute survives to the next episode.

**Sandbox.** The process can read only the Python install and your own file: no data, no files, no environment,
no network, no new processes or threads, no writes. Printing goes to the run log (keep it short).
