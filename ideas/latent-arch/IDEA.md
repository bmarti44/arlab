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

## Task: frozen synthetic state tracking (generator in `frozen/prepare/`, PREPARE-only)

**v2 (2026-10-01).** The v1 arithmetic programs (below, in git history) were not learnable at pack scale: the
baseline stayed at chance after three attempts (BLOCKED.md B2). gpt-6.1-sol's researched fix (`FIX-sol.md`) replaced
them with ordered permutation composition plus dense supervision through targets only.

**Records.** `BOS = s0 t1 … t14 ?` (18 tokens). s0 is a start state in {0..4}. Exactly k of the 14 slots hold an
active operator, one of the **44 fixed-point-free permutations** of the 5 states (one token each, `10`…`53`), at
uniformly random positions. The other slots hold the no-op `-`. The answer is s0 pushed through the active operators
in order, one state token predicted at `?`. **Depth k** = the number of active operators = serial composition steps.
The length is fixed and active positions are uniform at every k. The no-op count does reveal k; this is accepted.
- **Splits.** ID k 1..6, DEPTH k 7..10, EXT k 11..12 (reported only). Validation and holdout each have 2,000 ID +
  2,000 DEPTH + 500 EXT items, plus 500 counterfactual twins (same operators, other s0; the answer always changes,
  since the composition is a bijection). Eval records are hash-disjoint across splits. Training records may repeat
  (k = 1 has only 3,080 distinct prompts), but no training row equals any eval prompt.
- **Where the data lives.** `train/programs_k{1..6}.bin` and `train/targets_k{1..6}.bin` (300 k records per k,
  uint16, 18 per record; target 65535 = ignore) are the only record data RUN sees. Eval records and answers are in
  `private/` only. `public/` is empty.
- **Secret vocabulary relabelling** is unchanged: all token data and `token_bytes` are permuted with a salt that
  exists only in `private/`.

**v2.1 (2026-10-01, second sol consult, `FIX-sol-v2.1.md`).** The v2 pilot learned the task: k 1–3 at 100 %,
then chance. But ID mastery timing varied by seed (acc_id 0.70 / 0.33 / 0.47), so σ of the old ID/DEPTH mean was
≈ 0.09, far above MES 0.04. acc_depth sat at chance with σ 0.009. sol's fix, adopted as is:
- **primary = acc_depth**: the equal-k mean over k 7..10, with 2,000 records per k (8,000 per split). acc_id becomes
  a guard;
- **660 s** training;
- **uniform k 1..6** throughout, with no curriculum;
- **standalone records** (896 × 18 tokens per batch; no packing, so no cross-record attention);
- baseline loss text CE + 0.125·final CE + 0.125·per-record prefix CE (surface-editable);
- LR constant to 80 %, then linear to 0.

**Training stream (frozen in the harness).** Each step gets `(x, y, xp, yp)`:
- **48×1024** climbmix text windows with next-token targets;
- **896 standalone records** (the token count of 16 rows), each `BOS = s0 t1 … t14 ?`;
- their targets: the state after each active operator, at that operator's position, and the final state at `?`.
  All other targets are -1. Intermediate states never appear in inputs.

**Expected outcome (sol's estimates, unvalidated):** baseline ID 0.75–0.95, depth 0.20–0.25, σ_depth 0.01–0.02.
**Fallback** if every arm stays at depth chance: train k 1..3 and evaluate depth on k 4..7.

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
3. **Heuristic floors** computed by the evaluator on the same items (v2): only the last operator applied to s0, s0
   itself, only the first operator, the modal answer. Chance is 20 %; the first/last-operator floors are ≈ 27 %
   because they are exact at k = 1.
4. **Counterfactual consistency — changed-answer root interventions** (50 pairs per k 1..10 per split): the same
   operators with a different start state s0 (v2; the answer always changes). Reported as the fraction of pairs where both answers are right.

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

## As built / known limits (v0 build + hardening after three astra reviews)

- **Programs.** Every program has exactly 2 constants (statements 0 and 1) + 12 operations (83 prompt tokens, 84
  with the answer), so length carries no depth signal. The queried statement's index is uniform on 11..13 for every
  k (the chain's last link sits right before it; other chain links at random earlier positions), so its position
  does not reveal k: per-k total-variation distance of the query index ≤ 0.05 across k 1..10 (sampling noise at 3000
  per k), ID vs DEPTH in the prepared data ≤ 0.04. **Every split**: no statement outside the queried chain is deeper
  than 6 and nothing reads the queried variable, so training and eval programs share the distractor distribution.
  Cues checked on 3000 programs per k (max pairwise TV across k 1..10): query index 0.02, chain-root index 0.03,
  query reads a variable vs a constant 0.03, number of `v=u±w` statements 0.04, answer value 0.10, refs to the query 0.
  Inherent, not matched: for k = 1 the query reads a constant statement directly; with fixed length, deeper chains
  leave fewer distractors, so distractor counts / references to the root / max distractor depth correlate with k
  (each is a weak cue and none identifies the answer). Pieces are single token ids; "79" uses `<|reserved_1|>`
  (`info.json: piece_fallback`). Counterfactual twins: 50 random originals per k in 1..10, answer changed.
- **Where secrets live.** The vocabulary permutation and tokenizer are in `audit/` (read by the pack tests only; never
  mounted into RUN or EVALUATE). There is no `public/`.
- **RUN.** `frozen/run/harness.py` is a supervisor that never imports the surface: it starts `trainer.py` (the only
  process with surface code), starts CLOCK_MONOTONIC after the trainer has loaded torch and the data, kills the
  trainer's process group at budget + 30 s, is a subreaper (kills every leftover descendant before writing), and
  alone writes `budget.json` and `stats.json`. The checkpoint is written by frozen trainer code as a flat
  `{name: tensor}` dict of every parameter and buffer of `state["model"]`; the supervisor validates it
  (`weights_only` load, dense tensors only) and records its sha256, tensor hash and element count.
- **EVALUATE.** `evaluate.py` never imports surface code. The surface's `make_model(config)` + the checkpoint run in
  `frozen/run/worker.py`, a separate process under Landlock (readable: Python install, /usr, /etc, /proc, /sys, the
  worker dir and /work; read/write: /dev for the GPU and a private scratch dir; no eval data, evaluator code or
  result dir; no TCP). It receives token ids as bytes and returns float32 logits via a memfd; labels, scoring and
  the result file stay in the evaluator. Model state: after loading, the worker reports the hash of every
  parameter/buffer and an inventory of tensors (and arrays/bytes > 1 MB) reachable from the model's module
  attributes and the surface's modules, globals, closures and classes; the evaluator requires the hash to equal the
  supervisor's checkpoint hash and the inventory to be empty (views of registered storage allowed), before scoring
  and again after it (so `eval()`/`train()` overrides or forwards that modify weights, and caches, are caught).
  Every byte of every storage behind a registered tensor must be covered by registered contiguous tensors (a
  buffer registered as `backing[:1]` with `backing[1:]` kept as a cache is invalid). Threads and processes: before
  the surface is imported the worker warms torch's own thread pools, records the OS thread set and disables every
  Python-level thread/process start (`_thread`, `threading`, `subprocess`, `_posixsubprocess`, `os.fork/spawn/
  system/popen`); after every forward and at each state check, any new OS thread (`/proc/self/task`) or child
  process makes the run invalid — the metering dispatch modes are thread-local.
  Warm-up uses random tokens only; every held-out prompt is sent only in the metered, audited scoring pass.
  FLOPs: matmul/attention/conv/FFT by formula (torch's registry plus `_addmm_activation`, `addbmm`, `mv`,
  `addmv`, `addr`, `dot`, `_int_mm`, CPU flash SDPA, FFTs); ops tagged pointwise/reduction and an explicit list of
  normalization/softmax/data-movement/scan/sort ops are charged max(elements read, written), so broadcast-multiply +
  sum costs about what the matmul does; views and factories are free; **any other op that reads tensors is
  rejected** (`unpriced:<op>`, run invalid). Extending the formulas or the allowlist is a pack change. `infer_flops_tok` = mean of the per-token counts on text and programs (actual shapes).
  Causality: for 4 text rows, 16 cut points per row, one drawn at random (os.urandom) in each of 16 equal strata of
  positions 0..1022, so every region of the row, including the last 64 positions, is probed on every evaluation;
  the future after the cut is replaced by random tokens and re-scored in a *fresh* worker, invalid if any logit at
  or before the cut moves by > 1e-3. Program rows are not re-checked: the answer is never in the input.
  `params_m` = float elements + bytes of non-float tensors (/1e6); `params_bytes` = total bytes (/1e6), both
  guarded at 1.05× (baseline ≈ 27.7 M elements: 26.35 M parameters + 1.31 M rotary buffers).
- **Known limits (not enforced by code; bounded by program.md and by the orchestrator's review of every kept or
  finalized diff).** (1) Kernels launched outside the PyTorch dispatcher (raw Triton/CUDA launches) are invisible
  to the FLOP counter and op audit; `infer_s` is the backstop. (2) The FLOP counter, op audit, state hash and
  inventory run inside the worker process, next to the surface, which could in principle tamper with them. (3) A
  symbolic interpreter inside the surface (relabelling is obfuscation: the token roles are learnable from training
  data) is forbidden by program.md, not detected. (4) `make_model()` runs in EVALUATE with a 120 s limit and no data
  access; it could still do data-free precomputation. (5) Landlock + CUDA in the worker, and the strict OS-thread check
  on CUDA (it assumes CUDA/cuBLAS create no new threads after the pre-import warm-up), are tested on CPU only until
  the GPU PROBE; a baseline flagged for threads there means the warm-up must be widened, not the check dropped. (6) Causality is sampled (64 cut points per evaluation, fresh random positions each time), not
  exhaustive: a surface that peeks only at a few specific positions is caught only with probability ~ (positions
  hit by some cut) — it would have to be tuned to the eval format, which program.md forbids. (7) The inventory does
  not see state in Python lists/scalars, in modules outside the surface, or monkey-patched onto torch itself;
  element counting can be gamed by packing into float dtypes, bounded by the bytes guard (same bytes = same
  information). (8) Elementwise FLOPs are a lower bound (e.g. exp counts 1 per element). (9) Native threads started
  through ctypes (`pthread_create`) that finish within a single forward leave no trace in `/proc/self/task`; only
  threads still alive after a forward are caught (ctypes-level tricks are the same class as limit 2).
