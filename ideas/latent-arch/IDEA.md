# latent-arch — tiny GPTs trained from scratch that compute in latent depth (v0, approved plan T4.2)

**The owner's interest.** Latent reasoning and new architectures: models that spend extra computation per token in
hidden states (loops, recursion, adaptive depth, continuous thoughts, recurrent/linear-attention state) instead of
written reasoning. `looped-latent` v0 tested a looped *retrofit* of pretrained Qwen3-0.6B and came out
**not_found_at_this_scale** (20 experiments, baseline 0.6215, no keep; holdout bound d+2SE = 0.025 < MES 0.03;
`~/arlab-runs/looped-latent/v0/report.md`). Every published retrofit that gained used ≥ 1 B tokens; ours saw ~5 M.
This pack removes the retrofit caveat: every arm is **trained from scratch** at nanochat-lite scale, under the same
parameter cap and the same wall-clock training budget, so the architecture is the only difference that can matter.

## Hypothesis and question

**H1.** Within a fixed parameter count (≤ 1.05× a 6-layer, 384-wide GPT) and a fixed wall-clock training budget,
some architecture/recipe on the surface (loops with input injection, adaptive depth, recursion, hybrid linear
attention, continuous-thought positions, …) answers held-out multi-step programs better than the plain GPT, **with no
written reasoning**, including programs with **more steps than any training program**, while staying a language model
(text val_bpb within 3 %) and paying for its extra per-token compute inside the budget.
**Null result is informative.** If nothing beats the plain GPT by ≥ MES, "latent depth does not pay at 26 M params /
~85 M tokens / 5 min" is a result (the looped-latent verdict, now without the retrofit excuse).

## Background (sources; numbers as reported, none reproduced)

- **Depth recurrence from scratch.** Universal Transformer (1807.03819). Huginn-3.5B, prelude/core/coda with input
  injection and random recurrence, 800 B tokens: GSM8K rises with test-time recurrence, two runs collapsed
  (Geiping+ 2502.05171). Saunshi+ (2502.17416): a k-layer block looped L times nearly matches a kL-layer model on
  reasoning, worse on memorization — **the text-bpb guard below measures exactly that trade-off.** Ouro/LoopLM
  (2510.25741): loops help knowledge *manipulation*, not capacity; 8 loops unstable, cut to 4.
- **Depth / length generalization on synthetic tasks.** Looped transformers with an adaptive number of steps
  length-generalize by running more loops at test time (Fan+ 2409.15647, ICLR'25). Recurrent-depth transformers
  trained to 5-hop implicit reasoning extrapolate to 10-hop with more iterations, limited by "overthinking" (Kohli+
  2604.07822). Extrapolation is **highly unstable across runs** at similar in-distribution accuracy; random loop counts
  in training reduce the OOD variance (Kuo+ 2606.29983) — expect large seed noise on the depth split (see MES).
- **Adaptive depth / recursion.** Mixture-of-Recursions: a token router picks recursion depth, lower perplexity at
  equal FLOPs, 135 M–1.7 B (2507.10524). TRM: a 2-layer, 7 M-param network recursed deeply beats HRM on
  Sudoku/Maze/ARC (2510.04871) — tiny recursive nets can win on algorithmic tasks, though not as language models.
- **Hybrid linear attention.** Gated DeltaNet (2412.06464), Kimi Linear (2510.26692); 3:1–6:1 linear:full ratios best
  for recall (2507.06457); associative recall explains most of the linear-vs-attention gap (Zoology 2312.04927).
  Theory: fixed-depth transformers and diagonal SSMs are in TC⁰ and cannot track state serially (Merrill+ 2404.08819);
  negative eigenvalues / non-diagonal transitions unlock state tracking in linear RNNs (Grazzi+ 2411.12537,
  DeltaProduct 2502.10297). Dissent: MiniMax dropped hybrids for multi-hop deficits at scale (memory-architecture
  RESEARCH.md §2).
- **Continuous thought.** Coconut (2412.06769) beats CoT only on search-like synthetic tasks; its latent tokens act as
  placeholders that promote shortcuts (2512.21711); "Soft tokens, hard truths" (2509.19170). Pause tokens help only
  when used in pretraining (2310.02226) — which this pack does.
- Predecessor brief with the full methods table: `ideas/looped-latent/RESEARCH.md`.

## Task: frozen synthetic programs (generator in `frozen/prepare/`, PREPARE-only)

**Programs.** Straight-line single-assignment programs over lowercase single-letter variables, arithmetic mod 100,
e.g. `a=37;k=12;b=a+4;c=k-3;d=b+c;…;e?` → answer `41`. Every statement is `v=CONST`, `v=u±c` or `v=u±w`
(c ∈ 1..9). **Depth k** = the length of the longest dependency chain from the queried variable to constants (the
number of serial arithmetic steps needed). **Every program has the same number of statements (N_STMT = 14)**; the
queried chain is interleaved with distractor chains, so depth never correlates with program length or position
(no length-generalization confound). The answer is the executed value, always **one token** (PREPARE asserts that
each of 0..99 and each variable letter tokenizes to exactly one token in context, and that the query line ends right
before the answer token).
- **Splits** (each from its own generator seed family, hash-deduplicated across splits on the normalized program):
  train ~2 M programs, queried depth k ∈ 1..6; validation and holdout each **2,000 ID items (k 1..6, uniform)
  + 2,000 DEPTH items (k 7..10, uniform)**. No training program is ever *queried* at k > 6.
- **Where the data lives.** `train/programs.bin` (token ids) is the only program data RUN sees. **Validation and
  holdout programs and answers are in `private/` only** (the evaluator runs the model on them itself), so RUN never
  sees an eval program, not even unlabelled. `public/` is empty.
- **Secret vocabulary relabelling.** PREPARE draws a random permutation of the 8192-token vocabulary (salt from
  `os.urandom`, stored only in `private/`) and writes *all* token data (text stream, programs, eval rows) relabelled;
  the evaluator's `token_bytes` are permuted to match. A general architecture is invariant to this (embeddings are
  learned); a surface that hard-codes token ids (the delimiter, digits, the query mark) to special-case the program
  format silently breaks.
- **Knobs** (`DIFFICULTY`, N_STMT, the train/ID/DEPTH ranges, the program share of the stream) are tuned **once in the
  GPU pilot** so the baseline lands at 30–70 % ID accuracy, then frozen; changing them changes `data_hash` (new tag).

**Training stream (frozen in the harness).** Each 64×1024 batch = 58 rows of random windows of the nanochat-lite
climbmix text (same pinned shards and tokenizer recipe) + **6 rows of packed programs** (BOS-separated, ~9 programs
per row; ≈ 9 % of tokens, ≈ 80 k programs per baseline run). Plain next-token loss on every token; the surface sees
only `(x, y)` and cannot tell rows apart except by learning to.

## Metric

**Primary: `accuracy`** = exact match of the argmax over the **full vocabulary** at the answer position, one forward
pass of `BOS + program + query` with **no generated tokens** (no room for written reasoning), over 4,000 items =
equal-weight mean of **acc_id** (k 1..6) and **acc_depth** (k 7..10). `items` = one 0/1 per program (paired SE).
Also reported, not gating: accuracy per k (1..10), acc on k = 11–12 (500 extra items per split), and the controls below.

**Memorization vs computation (how the eval tells them apart).**
1. **The depth split cannot be memorized**: no training answer was ever supervised at k ≥ 7, and program length is
   fixed, so only a model that composes steps can answer. A lookup/heuristic model shows a cliff at k = 7; a computing
   model degrades smoothly. The accuracy-vs-k curve is in every report.
2. **Novel instances**: random constants make the program space ≈ 10²⁰; eval programs are hash-disjoint from train.
3. **Heuristic floors** computed by the evaluator on the same items: last constant in the program, the queried
   variable's own constant term, the chain root, the modal answer. The generator is tuned so each is ≤ 10 %.
4. **Counterfactual consistency** (500 pairs per split): the same program with the chain-root constant shifted by δ;
   the answer shifts by ±δ (sign fixed by the chain). Reported as the fraction of pairs where both answers are right.

## Budget — how loops, recursion and extra positions are paid for

- **Training compute = wall clock (binding).** The frozen harness starts its timer **before importing the surface**
  and stops handing out batches at `--train-seconds 330` (import, `build`, `torch.compile` and every `train_step`
  count; it synchronizes CUDA before each deadline check). Anything that costs more per token — K passes through a
  shared block, extra latent positions, deeper recursion, a slower pure-PyTorch linear-attention scan — sees
  proportionally fewer tokens. Nothing inside the training budget is free. `budget.json = {"train_seconds": t}`,
  `budget.limit = 360` (a final step may overrun by ≤ 30 s; beyond that the run is invalid). The surface gets
  `progress = elapsed / budget` (not `total_steps`, which is unknowable) for its schedules.
- **Inference compute = counted FLOPs (binding) + measured time (backstop).** The evaluator loads the checkpoint
  eagerly and runs under `torch.utils.flop_counter.FlopCounterMode` plus a `TorchDispatchMode` op logger on a fixed
  probe (8 text rows × 1024 and 256 programs of all depths): `infer_flops_tok` = counted matmul/attention FLOPs per
  scored token on the *actual* inputs, so data-dependent halting, per-token recursion depth and extra positions are
  all counted where they happen. Guard: **≤ 2.0× the baseline** (the 2×-depth reference's cost). Averaged over the
  probe, so adaptive depth may spend more on deep items and less on shallow ones. Any op outside the `aten`/`prims`
  namespaces (custom Triton/CUDA kernels, which FlopCounterMode cannot see) makes the run **invalid**;
  `infer_s` (timed forward over the program eval) ≤ 2.5× baseline catches anything the counter still misses.
- **Parameters (binding).** `params_m` = parameters + floating-point buffers of the loaded model, counted by the
  evaluator, ≤ 1.05× baseline (26.0 M incl. embeddings and value embeddings). A loop reuses weights, so it is
  compared with the plain GPT at equal *parameters* and equal *training time*; the 2×-depth reference breaks the
  parameter cap on purpose and is reported only.

**Per-run time (estimate from nanochat-lite, idle GB10).** nanochat-lite trains 62.9 M tokens in ~225 s
(~285 k tok/s after compile; RUN ~4 min; median 5.2 min per experiment incl. agent in m3b). Here: 330 s budget ≈
~25 s compile + ~305 s × 285 k ≈ **~85 M tokens for the baseline** (≈ 77 M text + 8 M program tokens); a K = 2 loop of
the whole stack sees ~45 M. EVALUATE ≈ 1.5 min (bpb on 2048×1024 rows, 5,000 program forwards, FLOP probe,
causality). **≈ 7.5 min RUN + EVALUATE per trial** → a 32-node tree round ≈ 4–4.5 GPU-hours (nodes run serially under
the GPU lock; proposals overlap with `--workers 4`); CALIBRATE (5 seeds + 1 reference) ≈ 45 min; FINALIZE (top-3 × 2
confirm seeds + 3 holdout seeds × 2 arms) ≈ 1.5 h. Re-measure in the pilot; `train_s` 1.3× ratio guard as usual.

## Surface, baseline, references

**Surface (what Codex may change): `surface/model.py` + `surface/train.py`** — the model (any causal architecture in
pure PyTorch: prelude/core/coda loops with input injection, random-K training, truncated BPTT, adaptive halting or a
recursion router, depth-wise LoRA on shared blocks, gated-delta / linear-attention layers mixed with full attention,
learned pause or continuous-thought positions *inside* `forward`, deep supervision) and its training recipe
(optimizer, LR, schedule over `progress`, batch accumulation, loss shaping that does not key on token ids).
API: `build(config) -> state`, `train_step(state, (x, y), step, progress) -> loss`, `save(state, path)`,
`load(path, device) -> model` whose `forward(idx)` returns causal logits `(B, T, V)` for the same `T`.
**Frozen:** data, mixture, batch shape (64×1024), vocabulary, the timer, the eval, all measurements.

**Baseline = the plain GPT**: nanochat-lite's m3b keep (6 layers × 384, 3 heads, value embeddings, MuonAdamW,
2^16 tokens per optimizer step, 26 M params) split into `model.py`/`train.py`, schedules re-expressed over `progress`.
The tree root is this baseline. **References (frozen/run/ref_*, screen seed only, reported, not the verdict's target):**
- `depth12` — the same GPT at 12 layers × 384 (~2× non-embedding params and FLOPs/token), same 330 s. The natural
  "is it just depth?" yardstick: a loop that matches it at half the parameters is itself a finding.
- Pilot only (not in pack.yaml; ~12 min each): `depth12_2x` (12 layers, 660 s: compute-unmatched ceiling) and
  `loop_naive` (6 unique layers looped 2×, additive input injection, random K ∈ {1,2,3} in training), to show whether
  the depth split moves at all for *some* architecture before any search is spent.

## MES, guards, power

**MES = 0.04** (4 points of the ID/DEPTH mean, e.g. +8 points on the depth split alone), fixed before calibration.
Item pack, 4,000 holdout items, 3 holdout seeds: SE = √(0.2/4000 + 2σ²/3).
σ = 0.010 → SE 0.0108 (2SE 0.022); σ = 0.015 → 0.0141 (0.028); σ = 0.020 → 0.0178 (0.036).
**Powered (2SE ≤ 0.04) iff σ ≤ 0.023**; the recommended margin (2.5 SE ≤ 0.04) holds iff σ ≤ 0.0176. Seed noise
dominates (from-scratch training + unstable extrapolation, Kuo+ 2606.29983), so more items barely help: if CALIBRATE
measures σ > 0.023, the fix is more holdout seeds (new tag, e.g. 101–105), never a larger MES. Tree screens are one
seed: a node ~2–2.5 σ_screen above the root at MES; FINALIZE follows TREE-SEARCH.md (top 3 → confirm → one holdout).
Seeds: calibration [1–5], screen 1, confirm [2, 3], holdout [101, 102, 103].

**Guards** (all computed by the frozen evaluator/runner):
- `val_bpb` ≤ **1.03×** baseline (text on the nanochat-lite validation rows, relabelled). A policy threshold, not a
  noise one (σ_bpb ≈ 0.003 ≈ 0.25 %): the model must stay a language model; ~3 % is about what the plain GPT loses
  from training on ~25 % fewer tokens (pilot measures and records it in DECISIONS.md before CALIBRATE).
- `acc_id` ≥ 0.90× baseline — a keep may not buy depth accuracy by breaking in-distribution accuracy.
- `params_m` ≤ 1.05×, `infer_flops_tok` ≤ 2.0×, `infer_s` ≤ 2.5×, `train_s` ≤ 1.3× (all vs baseline);
  `peak_mem_gb` ≤ 40 (the GPU is shared with the owner's jobs).
- **Invalid:** non-finite loss/logits; wrong logit shape; non-causal model (`arlab.lib.lm.causality_check` on text
  *and* program rows); non-`aten` ops in the probe; `train_seconds` > 360; weights changed after the deadline
  (hash at `save` vs hash at load).

## Anti-gaming

- **Frozen generator and eval.** `progen.py`, seeds and `prepare.py` live in `frozen/prepare/` (PREPARE only); the
  evaluator and every eval program live in `frozen/eval/` and `private/`, never mounted in RUN. RUN cannot regenerate
  labels, cannot see holdout seeds, cannot see any validation or holdout program.
- **No special-casing of the format.** Rule in `program.md`: the architecture may not branch on specific token ids or
  hand-code knowledge of the program syntax; adaptive compute must be learned from content. Enforced by the secret
  vocabulary relabelling (hard-coded ids point at random tokens) plus diff review of every node (`diff.patch`).
- **No home-made training data.** The surface must train only on the batches the harness hands it (no synthetic
  program generation, no reading files/env/network — same rule as nanochat-lite). A surface that generated deep
  programs would defeat the depth split; relabelling makes it hard (it would have to infer the token map), review
  catches it, and the depth split falls apart if it happens anyway (training-free generalization is what is measured).
- **Scores from raw logits only.** The evaluator never trusts a surface number; FLOPs, params, time and causality are
  measured on the loaded model object that is scored.
- Residual (documented, as in every in-process training pack): the surface runs inside the harness and could in
  principle patch the timer or introspect the harness; Codex is a good-faith proposer and diffs are in the report.

## Honest caveats and expected outcome

- **Tiny scale.** 26 M params, ~85 M tokens, 5.5 min. Huginn/Ouro used 0.8–7.7 T tokens; nothing predicts whether
  loop benefits appear at 10⁻⁴ of that. The synthetic program share (~9 %) is small; a win may reflect how the
  architecture learns *this* task in 8 M tokens rather than a general reasoning gain.
- **Synthetic task.** Single-assignment arithmetic chains test multi-hop composition, not reassignment/state tracking
  (Merrill 2404.08819) or anything agentic. Positive results say "latent depth helps serial composition at this scale".
- **Recipe wins.** The surface includes the recipe, so a keep can be an LR/schedule change that happens to favour the
  program data. The root is the tuned m3b recipe and every keep's tag and diff are in the report; label such wins as
  such. **Linear attention is handicapped**: no `fla` kernels (custom kernels are banned by the FLOP audit; sm_121
  support is doubtful anyway), so pure-PyTorch chunked scans pay a wall-clock penalty at T = 1024 where their
  asymptotic advantage does not show.
- **Wall clock on a shared GPU.** Throughput changes shift token counts; contended runs are re-run (runner GPU wait),
  σ absorbs the rest. Compile time counts, which penalizes architectures with many recompiles (e.g. variable K).
- **Depth-split floor.** A 6-layer plain GPT likely scores near chance (1 %) at k ≥ 7; if *no* pilot architecture
  (depth12_2x, loop_naive) moves it, shorten the DEPTH range (e.g. 7–8) in the pilot, before calibration.
- **Prior.** ~35–45 % that some keep clears +0.04 under the 1.03× text guard; the likeliest failure is that loops win on
  programs but lose > 3 % text bpb from fewer training tokens (Saunshi's reasoning-vs-memorization trade-off).

## Build notes (files the pack needs)

- `pack.yaml` — name `latent-arch`; image `nvcr.io/nvidia/pytorch:25.10-py3`; prepare `python /prepare/prepare.py
  --out /data` (timeout 3600); run `python /frozen/harness.py --out /out --seed {seed} --split {split}
  --train-seconds 330` (timeout 600, gpu, mem_gb auto); evaluate `python /eval/evaluate.py --run /run_out --out
  /result/metrics.json` (timeout 300); `budget: {unit: train_seconds, limit: 360}`;
  `metric: {name: accuracy, direction: maximize, mes: 0.04}`; the guards above; seeds as above;
  `references: [{name: depth12, path: frozen/run/ref_depth12}]`; agent gpt-6-sol high;
  campaign sized for tree rounds of 32.
- `frozen/prepare/prepare.py` — nanochat-lite's climbmix download + tokenizer (same pins), secret vocabulary
  permutation, relabelled `train/tokens.bin` and `train/programs.bin`, text eval rows + permuted `token_bytes` and
  program items/answers/k/controls into `{validation,holdout}/private/`, `splits.json {"validation": 4000,
  "holdout": 4000}`; single-token assertions. `frozen/prepare/progen.py` — seeded generator, depth k, fixed N_STMT,
  hash dedup, `DIFFICULTY`.
- `frozen/run/harness.py` — timer before surface import; fixed 58 text + 6 program rows per batch (seeded order);
  CUDA-synced deadline check; `progress`; NaN stop; `save`; writes `budget.json` and `stats.json` (tokens seen, steps,
  compile/first-step time, weight hash at save). `frozen/run/ref_depth12/{model.py,train.py}`.
- `frozen/eval/evaluate.py` — load eagerly; weight hash; params+buffers; FLOP counter + op-namespace audit on the probe;
  causality on text and programs; `val_bpb` (from logits, permuted `token_bytes`); program accuracy at the answer
  position (full-vocab argmax), per-k, heuristic floors, counterfactual pairs, timed `infer_s`; `items`; `valid:false`
  on NaN/shape/causality/op-audit/hash failures.
- `surface/model.py` + `surface/train.py` — the m3b baseline split in two, schedules over `progress`.
- `program.md` (≤ 80 lines): goal, API, the budget/FLOP/param rules, the no-token-id and no-synthetic-data rules,
  ideas list (loops + injection, random K, halting/MoR, GDN hybrid in pure PyTorch, pause/latent positions).
- `requirements.txt` — none beyond nanochat-lite's (rustbpe, tiktoken, pyarrow, requests).
- `tests/` — generator determinism and split disjointness (hash), depth k computed correctly and uncorrelated with
  length, single-token answers, heuristic floors ≤ 10 % on validation, scorer (argmax at the right position, off-by-one
  rejected), FLOP counter counts a 2× Python loop as 2× and flags a non-`aten` op, causality check rejects a
  future-peeking model, relabelling round-trip, a surface that hard-codes a token id loses accuracy under relabelling,
  harness deadline stops a slow `train_step` and marks > 360 s invalid.
