# ttc-controller — instructions for the research agent
You make ONE change per call. The runner (not you) runs the experiment, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator or the data; do not try.
## Goal
maximize accuracy: held-out GSM8K exact match of the answer your controller picks from a fixed cache of 32 sampled
reasoning traces per problem (Qwen3-0.6B, non-thinking, ~235 tokens each), under a pooled budget of B generated
tokens per problem on average. Your controller decides how many traces to read, how far to read each one, when
to stop, how to pace the pool across problems, and how to aggregate. The verdict compares with the baseline
(majority vote over whole traces at the fair share, about 4 traces per problem at B).
Current noise: see sigma in history.md. A change is kept only if it beats the incumbent on one seed and then
clearly (by more than 2·SE) on fresh seeds.
## What you may change
surface/controller.py only (the Controller class; numpy and the standard library are available). The API is in
frozen/run/api.md: read it. The harness, the trace orders, the budget accounting, the sandbox, the cache, the
scorer and the replicate protocol are frozen.
## What the code does
For each budget level (B, 0.5·B, 2·B) and each of 8 replicates the harness starts a fresh sandboxed process, builds
Controller(cfg), calls fit(train) with labelled train problems, then solve(problem) for every validation problem
in a seeded order. Reads are charged exactly (in 32-token chunks) against the pool; problems after the pool runs
out score 0. You see per-token signals (logprob, DeepConf confidence, top-5 entropy) of what you read, and an opaque
answer label once a trace is read to its end. The evaluator replays the harness's read log and scores the labels.
## Guards (runs that fail one are discarded)
- acc_half and acc_double (accuracy at 0.5·B and 2·B) ≥ 0.97 × the baseline's: B arrives in cfg; never hard-code it.
- RUN wall time (24 episodes) ≤ 600 s; fit() + __init__ ≤ 60 s and solve() ≤ 5 s per call, or the run crashes.
- budget_fraction ≤ 1 (the harness enforces it). A label not produced by a trace read to its end, or a broken
  sandbox, makes the run invalid.
## Forbidden (bounded by these rules and the sandbox)
- Reading files, data, the environment or the network; starting processes or threads; trying to escape or probe
  the sandbox; talking to the harness except through the problem/trace objects.
- Keeping state across episodes or processes (each process is fresh; don't try to recognise problems).
- GSM8K-specific priors (answers are opaque labels on purpose).
## Ideas worth trying (from IDEA.md)
Early stopping when the first 2–3 finished traces agree (ESC windows) and spending the saved tokens on problems
that disagree; Adaptive-Consistency's Beta stopping rule with a cap; pacing that keeps a reserve for hard problems
late in the order; DeepConf filtering (drop traces whose lowest 64-token group confidence is low) and
confidence-weighted votes; aborting a trace early when its prefix confidence drops; interleaving width and depth
(open several traces, read 64–128 tokens of each, continue the confident ones); a small logistic vote weight or
stopping rule learned in fit() from the train traces' signals and correct flags; ignoring truncated traces.
## Rules
- One change, one hypothesis_tag (reuse an existing tag for the same idea).
- Read history.md: don't repeat a failed idea unless you change it materially. After 3 discards in a row, try something structurally different.
- Respect constraints.md (facts about this machine).
- Prefer simple changes; equal results with less code are better.
- Keep notes.md short and useful to your future self.
- Finish with the JSON object required by the output schema and nothing after it.
