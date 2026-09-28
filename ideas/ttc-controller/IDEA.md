# ttc-controller — a test-time-compute controller on cached reasoning traces (v0)

**Source.** Stage T4 of `docs/TREE-SEARCH.md`. This pack is built to be searched by `arlab tree`. Its RUN replays a
fixed cache of sampled reasoning traces on the CPU, so one trial takes seconds and a tree node costs about one Codex
call. That makes it the arlab setting closest to Dream-RSI (arXiv 2609.14858), where the object being improved is a
small program run in a replay environment. AutoTTS (arXiv 2605.08083) ran this loop with a coding agent: an explorer
LLM edited a width/depth controller replayed on 128 pre-collected Qwen3 traces per problem. On held-out AIME25/HMMT25
it kept SC@64's accuracy (45.3 vs 45.2 %) with 69.5 % fewer tokens, and the controllers transferred to other model
scales and to GPQA. Its accuracy gains at equal tokens were small and measured on 30-problem sets without error bars.
Prior work this pack builds on: self-consistency (2203.11171), Adaptive-Consistency (2305.11860), early-stopping
self-consistency ESC (2401.10480), self-certainty best-of-N (2502.18581), DeepConf (2508.15260).

**Question at this scale.** Take a fixed pool of sampled traces and a fixed mean budget of generated tokens per
problem. Does an agent-searched controller answer more held-out problems than majority voting at the same budget?
The controller decides how many traces to read, how far to read each one, when to stop, and how to aggregate.
It never sees any gold answer of a problem it is scored on. Prior that a keep clears the MES: about 40 %.
The main source of gains should be moving tokens from easy problems (early agreement) to hard ones, plus
confidence-weighted voting. See "Expected effect" below.

## Model, sampling and data

**Model.** Qwen3-1.7B, from the cached snapshot `70d244cc`, bf16, served with vLLM (`nvcr.io/nvidia/vllm:26.04-py3`)
in non-thinking mode (`enable_thinking=False`). The prompt is the question followed by "Please reason step by step,
and put your final answer within \boxed{}." Sampling uses Qwen's non-thinking settings: temperature 0.7, top_p 0.8,
top_k 20, `max_tokens` 1024, and `logprobs=5`. Each problem gets 32 samples (`n=32`, one request, shared prefix).

**Why non-thinking 1.7B, and not thinking mode or Qwen3.5-4B.** Thinking traces are about 5× longer, so the same
GPU time buys about 5× fewer problems (a holdout of about 260, which cannot resolve less than about 7 points).
Qwen3.5-4B runs at about 540 tok/s with 20 concurrent streams, so the cache below would take about 15 h, and GSM8K
would be closer to its ceiling. Non-thinking chain-of-thought is the regime where self-consistency was first shown
to help, and its short traces buy enough problems for power. What this costs us: DeepConf's largest gains were on
long thinking traces, which are not tested here (caveat 2).

**Data: GSM8K** (Cobbe et al. 2110.14168, MIT licence). The cached `openai/gsm8k` snapshot is pinned at `740312ad`.
Every answer is an integer, so grading is exact match after a frozen normalization (strip `,` `$` and spaces, and
`12.0` → `12`). Splits are by problem and fixed:
- *holdout*: the full GSM8K test set, 1,319 problems, used only in FINALIZE.
- *validation*: 1,000 problems drawn from GSM8K train with split seed 20260928.
- *train*: 700 more GSM8K-train problems, labelled. They are passed to the controller's optional `fit()`.
Before the split, any train-pool problem that near-duplicates a test problem is dropped (8-gram Jaccard > 0.8 on the
normalized question).

**Deviation from the brief (flag for review).** The controller may fit on *labelled train problems*. These are
disjoint problems, so it still never sees the gold answer of any problem it is scored on. The reason: learned
aggregation (for example a logistic map from confidence features to the weight of a vote) is a main family of
controllers. Without labels, the search could only hard-code constants tuned on validation. To go label-free,
remove the `correct` field from the train cache (new tag).

**Trace cache.** Per trace we store:
- per-token arrays in fp16: the sampled token's logprob; DeepConf's token confidence (the negative mean of the
  returned top-5 logprobs); and the entropy of the renormalized top-5;
- the length, the finish reason (`stop` or `length`), and the extracted answer (the last `\boxed{}`, normalized;
  `null` if the trace was truncated or has no box);
- the text, which goes to `private/` only and is used for auditing and the scorer tests.

We add no intermediate "probe" answers, unlike AutoTTS: that would need extra generation per prefix. Prefix
information comes from the per-token signals instead (DeepConf-online style).

**Sampling happens outside arlab's PREPARE.** PREPARE has no GPU and no services, so the sampling runs as a separate
job before the build. The GPU job is `build/sample.sh`. It is launched detached (`systemd-run --user
--unit=arlab-ttc-sample`), holds `~/.cache/arlab/gpu.lock`, waits while the owner's jobs use the GPU, and writes
resumable 100-problem shards into `frozen/prepare/traces/` together with a `MANIFEST.json`. The manifest records the
SHA-256 of every shard, the model snapshot, the vLLM image digest and the sampling parameters. That folder is
git-ignored but part of `data_hash`, so the cache is pinned byte for byte. After that, arlab's (CPU) PREPARE checks
the manifest, applies the splits and writes the public caches and private gold. vLLM batching is not bitwise
deterministic, so the cache is the frozen artifact, not the sampler.

**PREPARE GPU time.** There are 3,019 problems × 32 traces × about 300 tokens (to be measured) ≈ 29 M generated
tokens. For Qwen3-1.7B at 128-way concurrency with `logprobs=5` we assume 1,500–2,500 tok/s: decode is limited by
memory bandwidth (3.4 GB of weights at 273 GB/s), and the known points are 540 tok/s for a 4B model at 20-way. That
gives **3.2–5.4 GPU h**, plus about 10 min of startup. The pilot measures throughput and mean length. If the
projection is over 6 h, cut *train* first (down to 400) and then *validation* (down to 800). Never cut the holdout.

**Pilot gate (100 train problems × 32 traces, about 10 min, before the full sampling).** The pilot must show:
- truncation (`length`) ≤ 3 %;
- no-box failures ≤ 2 %;
- pass@1 in [0.50, 0.88];
- maj@32 − pass@1 ≥ 0.05 (there is headroom to aggregate);
- throughput and mean length that project ≤ 6 h.

If pass@1 > 0.88, switch to Qwen3-0.6B (cached) before the full sampling, and record the switch in DECISIONS.md.
The pilot traces are kept as part of *train*, provided the sampling parameters do not change.

## RUN: the replay

**Unit and budget.** The budget unit is *generated tokens revealed from the cache*. B is the mean budget per problem.
The default is **B = 4 × L̄**, rounded to 50 tokens, where L̄ is the mean trace length on *train*. B is fixed in
`pack.yaml` (`--budget-tokens`) after the pilot and before calibration.

The budget is a **pooled** allowance, P = B × n_problems per replicate. Problems arrive one at a time in a seeded
order. The controller sees the pool left and the number of problems left, and it must pace itself. The harness
refuses every read beyond the pool, so any problem after the pool runs out scores 0. Prompt and prefill tokens are
not charged: with prefix caching they are nearly free, and they are identical for every controller.

**Replicates.** One run with seed s plays R = 8 replicates. Each replicate uses its own problem order and its own
per-problem permutation of the 32 traces, both derived by the harness from (s, r, problem). The controller can only
take the *next* trace in that order, so it cannot choose samples. Seeds therefore mean different orders and
subsets of the same 32 traces.

The item score is a problem's accuracy averaged over the 8 replicates at budget B. Every replicate at every budget
level starts a *fresh* controller process. This stops a controller from recognising a problem it met in an earlier
replicate (from its signal arrays) and reusing the traces it read there.

**What the controller sees** (the harness passes it objects, never files):
- `open()` returns the next trace.
- `trace.read(n)` reveals the next n tokens, rounded up to multiples of 32. Only revealed tokens are charged. It
  returns the three per-token signal arrays for those tokens.
- Once a trace has been read to its end, the controller also sees its length, its finish reason and its
  **answer label**. Labels are opaque per problem (`a0, a1, …` in the order they were first revealed, relabelled in
  every replicate). The controller never sees answer values. That rules out GSM8K-specific priors such as "prefer
  integers", which would not transfer.
- The controller never sees the problem text or the problem id, and never sees a trace it has not opened.
- `solve()` returns one label, or `None` (abstain, which scores 0).

**Isolation (pack-internal, about 60 lines).** The surface runs in a child process that speaks JSON lines with the
harness, the same pattern as tree-search policies. Before it imports `controller.py`, the child locks itself down
with an unprivileged **Landlock** ruleset: it may read only the Python install and `/work`, and it may not read
`/data`, `/frozen`, `/out` or `/proc`, nor write anywhere. The container has `--network none`.

At startup the child tries to open a cache file and `/proc/<ppid>/mem`. If either succeeds, the run is invalid. A
broken sandbox therefore shows up as `invalid` and never as a silent leak.

Limits: 5 s per `solve()` call, and 60 s for `fit()`.

If Landlock is refused under Docker's default seccomp profile (the builder verifies this in TESTS), the fallback is
the in-process risk every arlab pack accepts: a good-faith proposer, a prohibition in `program.md`, and the evaluator
audit below. Record that in DECISIONS.md.

## Metric, MES, guards

**Metric.** Held-out exact-match accuracy at budget B. It is an item pack with one score per problem (the mean over
8 replicates). The evaluator never trusts the controller or its log. It recomputes everything from the harness's
read log and the private gold. A run is **invalid** if:
- a chosen label was never produced by a trace read to its end in that replicate;
- the tokens charged differ from the sum of reads, or the reads exceed the pool;
- the answer count is wrong;
- the sandbox self-check failed.

**Secondary metrics** (reported; two are used as guards): accuracy at 0.5·B and at 2·B (R = 8 each); the mean
tokens used and the fraction of the pool used; the abstain rate; the mean number of traces opened; and accuracy
against each reference.

**MES (fixed before calibration): 0.03** (3 points of accuracy). The holdout is 1,319 problems × 3 seeds.

Averaging over replicates makes the seed-to-seed σ small. We estimate 0.002–0.005, because most problems are
decided the same way under every order.
- Conservative item formula: SE = √(0.2/1319 + 2σ²/3) = 0.0126 at σ = 0.003, so 2·SE = 0.025 ≤ 0.03. The pack is
  powered iff σ ≤ 0.0105.
- The runner uses the measured paired item sd against the references. At sd 0.25, SE = 0.0073 and 2.5·SE = 0.018. At
  sd 0.35, SE = 0.0099 and 2.5·SE = 0.025. So 0.03 meets the 2.5·SE sizing rule under any plausible sd.
- Why not 0.02: 0.02 is only powered through the measured paired sd, and the holdout cannot grow without mixing in
  train problems. Effects of 2 points will be reported with their CI but not called.

Seeds: calibration [1–5], screen 1, confirm [2, 3], holdout [101, 102, 103]. Runs are seconds long, so extra seeds
are free.

**Guards.**
- `acc_half` and `acc_double` ≥ 0.97× baseline. The controller receives B as a parameter, and a keep must not
  collapse at other budgets. This catches controllers hard-coded to one budget.
- `train_s` (RUN wall time) ≤ 600 s absolute, plus the per-call timeouts above. The controller's own compute must
  stay negligible next to generation.
- `budget.json` = `{"budget_fraction": max over replicates and budget levels of used / pool}`, limit 1.0. The harness
  already enforces this.

**Expected effect (rough).** With B = 4 traces, about 70 % easy problems settle after 2–3 agreeing traces. That frees
up to about 8 more traces for each hard problem. On the hard problems, maj@12 against maj@4 plausibly adds 5–10
points, which is +1.5–3 points overall. Confidence weighting or filtering could add about 1 more. The MES sits at the
top of that range, and that is why the prior is only about 40 %.

## Surface, baseline, references

**Surface: `surface/controller.py` only.** It defines a `Controller(cfg)` class, where `cfg` holds `budget`,
`n_problems`, `max_traces` = 32 and `seed` (for its own RNG). It has an optional `fit(train)`, where every train
problem carries its traces' full signals, labels and a `correct` flag per trace. And it has `solve(problem)`.

**The agent may change:**
- the stopping rule;
- the width/depth interleaving (how many traces are open and how far each one is read);
- aborting traces early from prefix confidence;
- pacing across problems;
- the aggregation rule (majority, weighted vote, filtering, best-of-n);
- what it learns in `fit()`.

numpy is available.

**It may not change:** the harness, the trace order, the budget accounting, the sandbox, the cache, the prompt or
the sampling, the scorer, or the replicate protocol. It may not read files or the environment, or keep state
between processes.

**Baseline (the tree root) = fixed-budget majority.** For each problem it takes the fair share
f = pool_left / problems_left. It reads whole traces in order until its spend reaches f (so the last trace can go
over, and later problems absorb the difference), and then takes a plurality vote with ties broken by first reveal.
At B, 0.5·B and 2·B this reads about 4, 2 and 8 traces. It is "majority@k at the same budget", with k set by the
budget and not hard-coded. `verdict: {compare_to: baseline}`.

**References (frozen, `frozen/run/ref_*`, report only).** All three use the baseline's reading rule unless noted:
- `conf_best`: best-of-n, taking the trace with the highest mean token confidence (self-certainty style);
- `conf_vote`: DeepConf-offline, which keeps the top 50 % of traces by lowest 64-token group confidence and takes a
  confidence-weighted vote;
- `asc`: Adaptive-Consistency's Beta stopping rule, capped at 3× the fair share, with the threshold set on *train*
  so that its mean spend ≈ B.

The report states whether a keep also beats `asc` and `conf_vote`. That answers "did the search find more than the
literature?" but it is not the verdict.

## Transfer check (secondary, never the verdict)

This runs after FINALIZE, and only if the campaign kept a controller other than the baseline. A second cache is
sampled from **Qwen3.5-2B** (cached `15852e8c`), a different generation with hybrid attention, in non-thinking
mode. If vLLM or its template fails, use Qwen3-4B (`1cfa9a72`). The cache covers the same 1,319 GSM8K-test problems
plus 300 train problems (for `fit()`), with 16 traces each. That is about 7.8 M tokens, or about 2 GPU h at an
assumed 1,000 tok/s, sampled by the same script into a separate directory. `frozen/run/transfer.py` replays the kept
controller, the baseline and the references with B' = 4 × L̄' and reports the paired d and the CI.

We expect a smaller gain. A reversal would mean the controller fits Qwen3-1.7B's confidence calibration rather than
a general allocation rule. A problem-set transfer (for example the SVAMP test set on Qwen3-1.7B, about 1 GPU h) is
the alternative if the second model will not serve.

## Known caveats

1. **Cache replay is not live sampling.** Traces are i.i.d. and fixed, and a controller can never read more than 32.
   It cannot condition generation: no branching from prefixes, no re-prompting with earlier answers, no "probe"
   answers. Cost is generated tokens only, and latency and batch efficiency are ignored. A replay win is a claim
   about *allocation and aggregation over i.i.d. samples*, nothing more.
2. **Small-model and short-trace ceiling.** This is a 1.7B model on GSM8K in non-thinking mode. Confidence signals in
   short chains of thought may behave differently from long thinking traces (DeepConf's regime). Absolute accuracy
   may be inflated by GSM8K contamination in Qwen's training data. The *relative* comparison is on identical traces
   and remains valid.
3. **Conditional on one cache.** Seeds re-order the same 32 traces, so σ does not reflect resampling the model. The
   claim generalizes over problems (the item SE), not over fresh model samples.
4. **Search overfitting.** A tree explores many nodes on one validation seed. The protections are the pre-registered
   tree FINALIZE (top 3 by screen, confirm on fresh seeds, one holdout run) and a holdout that is never touched
   before that.
5. **Small effect.** Published adaptive methods mostly *save tokens at equal accuracy* rather than add accuracy at
   equal tokens (AutoTTS, ASC, ESC). A fixed budget converts savings into accuracy only as far as the hard problems
   have headroom, and that is what the pilot's maj@32 − pass@1 gate checks.
6. **Landlock availability** under Docker's seccomp profile is unverified. See the fallback above.

## Build notes

- `pack.yaml`:
  - `name: ttc-controller`;
  - image `nvcr.io/nvidia/pytorch:25.10-py3` (numpy; CPU only);
  - `prepare: python /prepare/prepare.py --out /data`, 1800 s;
  - `run: python /frozen/harness.py --out /out --seed {seed} --split {split} --budget-tokens <B> --replicates 8`,
    with `gpu: false`, `mem_gb: 8` and `timeout_s: 900`;
  - `evaluate: python /eval/evaluate.py --run /run_out --out /result/metrics.json --budget-tokens <B>`;
  - `budget: {unit: budget_fraction, limit: 1.0}`;
  - `metric: {name: accuracy, direction: maximize, mes: 0.03}`;
  - guards `acc_half`/`acc_double` `min_ratio_vs_baseline: 0.97` and `train_s max: 600`;
  - seeds as above;
  - references `conf_best`, `conf_vote`, `asc`;
  - `verdict: {compare_to: baseline}`;
  - `agent: visible [program.md, IDEA.md, frozen/run/api.md]`;
  - campaign sized for a tree: `max_experiments` 200.
- `build/sample.sh` and `frozen/prepare/sample.py`: vLLM offline `LLM.generate` with `n=32` and `logprobs=5`, GPU lock,
  resumable shards and the manifest. The `--pilot 100` flag prints the gate table. The `--model` and `--out` flags are
  used for the transfer cache.
- `frozen/prepare/prepare.py`: loads GSM8K offline from `/hf`, dedups, splits, verifies the shard manifest and writes
  the following layout. `train/` holds the traces plus `correct` flags. `{validation,holdout}/public/traces.npz` holds
  the flat fp16 signal arrays, offsets, lengths, finish reasons and answers (no gold). `private/` holds the gold and
  the texts. It also writes `splits.json`. `frozen/prepare/traces/` is git-ignored.
- `frozen/run/`: `harness.py` (replicates, orders, pool, read log, budget.json, one child per replicate and budget
  level); `child.py` (Landlock, self-check, JSON-lines proxy objects); `api.md` (the controller API for the agent);
  `ref_conf_best/`, `ref_conf_vote/` and `ref_asc/` (each holds a `controller.py`); `transfer.py`.
- `frozen/eval/evaluate.py`: re-derives every score from the read log and the private gold, runs the validity checks
  above, and writes `items` per problem and the secondary metrics.
- `surface/controller.py`: the fixed-budget majority baseline (≤ 60 lines).
- `program.md`: the goal, the API, the guards, what is forbidden (file, environment or network access; state kept
  across replicates) and ideas worth trying (ESC windows, ASC stopping, DeepConf filtering, pacing, a learned vote
  weight).
- `tests/test_pack.py`:
  - the normalizer and answer extraction on hand-written cases (`\boxed{1,000}`, `12.0`, no box, two boxes);
  - that reads past the pool are refused;
  - that a label from an unfinished trace is rejected;
  - that the trace order depends only on (seed, replicate, problem);
  - that the sandboxed child cannot open `/data/validation/public/traces.npz`;
  - that the baseline's spend is ≤ the pool at all three budget levels;
  - that `evaluate.py` flags a forged read log as invalid.
