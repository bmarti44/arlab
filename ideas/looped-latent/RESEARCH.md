# Looped / latent reasoning on a small open model: research brief

Written 2026-09-27. Scope: can we retrofit "think in full-precision hidden states for several passes, then emit a token" onto a
Qwen3/Qwen3.5 model (0.6B–4B) on one GB10, and what is the smallest credible arlab experiment. Numbers are as reported by the
cited papers. I have not reproduced any of them.

## 1. The idea, and what "like Astra" can and cannot mean

The owner's idea covers two different mechanisms, and they should be kept apart:

- **Depth recurrence (looped transformer).** For each token, a middle block of layers runs K times on the residual stream
  (bf16 hidden state, never quantized to a token), then the coda and LM head emit one token. The sequence stays the same length.
  This is what "looped transformer" and "recurrent depth" mean in the literature (Huginn, Ouro, retrofitted recurrence).
- **Continuous-thought tokens (Coconut family).** The model emits no token. It feeds its last hidden state back as the next input
  embedding for n extra *sequence* positions ("latent thoughts"), then emits the answer. Each thought costs a full forward pass
  and adds a KV entry.

**GPT-6 Astra: no verifiable architecture detail exists.** The recurrent-depth claim comes from one pre-release report in
*The Information* (2026-09-01) that cites one anonymous source
([Raschka](https://magazine.sebastianraschka.com/p/gpt-6-astra-looped-transformers-and),
[kingy.ai summary](https://kingy.ai/blog/recurrent-depth-openai-astra/)). OpenAI's Astra system card and launch material do not
name recurrent depth, looped transformers or any shared-layer design. What OpenAI does state officially is that Astra's
chain-of-thought is shorter and less monitorable, and that the model is stronger at no-CoT tasks. OpenAI's chief scientist said
the "depth of the computation graph for our present frontier models, including Astra, is within a factor of two of GPT-4"
([Pachocki on X](https://x.com/merettm/status/2095023204993490967)). That statement neither confirms nor rules out a limited
(≤2×) loop. Background discussion:
[LessWrong](https://www.lesswrong.com/posts/PLisnSFir8y5AHkmP/how-concerned-should-we-be-about-astra-s-recurrent) and
[ACX "Specter of Neuralese"](https://www.astralcodexten.com/p/the-specter-of-neuralese).
**Treat "Astra is looped" as an unconfirmed rumor.** This brief relies only on the open literature.

## 2. Methods table

| Method | Mechanism | Size | Retrofit / scratch | Reported gain (as reported) | Code / weights |
|---|---|---|---|---|---|
| [Universal Transformer](https://arxiv.org/abs/1807.03819) (2018) | all layers shared, ACT halting | small | scratch | algorithmic/LM gains over vanilla | yes (tensor2tensor) |
| [Giannou+ 2023](https://arxiv.org/abs/2301.13196) | 13-layer looped TF as programmable computer | theory | — | expressivity proof only | — |
| [CoTFormer](https://arxiv.org/abs/2310.10845) | re-feeds tokens through the stack, CoT-like interleaving, adaptive depth | ≤~0.4B | scratch | beats same-depth looped baseline | [yes](https://github.com/epfml/CoTFormer) |
| [Saunshi+ 2025](https://arxiv.org/abs/2502.17416) | k layers × L loops ≈ kL-layer model; loop ≈ T CoT steps | ~1B | scratch | looped nearly matches deep model on reasoning, worse on memorization | — |
| [Huginn-3.5B](https://arxiv.org/abs/2502.05171) (Geiping+ 2025) | prelude 2 / core 4 / coda 2, random state init, input injection, r̄=32, TBPTT k=8 | 3.5B, 800B tok | scratch | GSM8K 0→34.8/42.1 (r=1→32), HumanEval 23.2 at r=32; two failed runs collapsed | [yes](https://huggingface.co/tomg-group-umd/huginn-0125) (cached locally) |
| [Ouro / LoopLM](https://arxiv.org/abs/2510.25741) (ByteDance 2025) | whole stack looped T=4, entropy-regularized exit gate | 1.4B/2.6B, 7.7T tok | scratch | 1.4B T=4: GSM8K 78.9, MATH500 82.4, HumanEval 74.4 (vs Qwen3-1.7B base 70.3/25.8/66.5); gain is knowledge *manipulation*, not capacity (~2 bits/param either way); 8 loops were unstable, cut to 4 | [yes](https://ouro-llm.github.io/) (2.6B cached locally) |
| [Relaxed Recursive TF](https://arxiv.org/abs/2410.20672) | tie layers of a pretrained model + depth-wise LoRA | Gemma 2B→1B etc. | retrofit (compression), multi-B-token uptrain | recursive 1B beats TinyLlama/Pythia, recovers most of the parent | partial |
| [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) | token-level router picks recursion depth, recursion-wise KV | 135M–1.7B | scratch | lower ppl / higher few-shot at equal FLOPs | [yes](https://github.com/raymin0223/mixture_of_recursions) |
| [Retrofitted recurrence](https://arxiv.org/abs/2511.07384) (McLeish+ 2025) | drop layers → (4,6,4) or (4,8,4); concat adapter 2h→h; noise init; recurrence curriculum 1→32; Muon | TinyLlama, OLMo-2-1B, Llama-3.2-1B | retrofit, 26–50B tok | TinyLlama GSM8K ~52% vs ~46% for the same-data non-recurrent baseline; beats post-training the plain model at equal FLOPs; "healing" phase needed | [yes](https://github.com/mcleish7/retrofitting-recurrence) |
| [Encode-Think-Decode](https://arxiv.org/abs/2510.07358) | loop 4 mid layers selected by angular distance + Kneedle (7-4k-5) | OLMo-2-1B | retrofit in mid-training, 50B tok | GSM8K 44.1→56.6 (k=5); MATH best at k=3; no code benchmarks | not found |
| [LoopUS](https://arxiv.org/html/2605.11011) (2026) | encoder/loop/decoder split, Mamba-style decay gate, sparse deep supervision, adaptive stop | Qwen3-1.7B/4B/8B, Phi-4 | retrofit, 3B tok FineWeb-Edu | Qwen3-4B ARC-C 50.4→54.0, avg +1.6–2.2; MMLU/HellaSwag flat; saturates at 3–4 loops | [project](https://thrillcrazyer.github.io/LoopUS) |
| [Training-free looped TF](https://arxiv.org/abs/2605.23872) (2026) | loop mid-4 layers of a frozen model with damped Euler/RK sub-steps; no training | Qwen3 0.6B–4B, 30B-A3B | inference only | naive re-looping hurts; damped: Qwen3-1.7B-Base 16-task avg +0.55 pp (window [12–15]), Qwen3-4B-Instruct MMLU-Pro +2.64; windows ≥6 layers hurt | not found |
| [Shapiro 2026](https://arxiv.org/html/2608.11233) | identity-preserving 1-loop path + gated bridge; full-block vs rank-16 LoRA | Qwen2.5-0.5B-Instruct | retrofit | depth extrapolation to ~1.5× trained; learned halting collapsed; loop-1 retention held | [yes](https://github.com/mshapiro123/recurrent-qwen-svgd) (single author, synthetic tasks) |
| [LoopCoder-v2](https://arxiv.org/abs/2606.18023) (2026) | parallel loop TF, R=2, shared KV + 64-token SWA for the loop | 7B, 18T tok | scratch | SWE-bench Verified 43.0→64.4 at 2 loops; ≥3 loops *regress* (27.6) | [weights](https://huggingface.co/Multilingual-Multimodal-NLP/LoopCoder-V2) |
| [Looped LMs for tool calling](https://arxiv.org/abs/2608.18171) (2026) | Ouro native vs McLeish-style retrofit, LoRA SFT on Hermes FC | 1–2.6B | both | retrofitted Llama-3.2-1B BFCL 21.4→32.9; native Ouro far ahead; gains concentrate on compositional/nested calls | no |
| [Coconut](https://arxiv.org/abs/2412.06769) (Hao+ 2024) | last hidden state fed back as next input; curriculum swaps CoT steps for latents | GPT-2 (≤Llama-3.2-1B) | fine-tune | GSM8K 34.1 vs CoT 42.9 (no-CoT 16.5); ProsQA 97.0 vs CoT 77.5; latents get homogeneous and collapse at larger scale | [yes](https://github.com/facebookresearch/coconut) |
| [CODI](https://arxiv.org/abs/2502.21074) | self-distill CoT teacher → 6 latents, hidden-state alignment | GPT-2, Llama-3.2-1B | fine-tune (LoRA r128) | GPT-2 ≈ CoT-SFT (43 vs 44); Llama-1B ~52 vs CoT-SFT ~58, no better than no-CoT SFT (~52); 48 h training | [yes](https://github.com/zhenyi4/codi) |
| [CoLaR](https://arxiv.org/abs/2505.16552) | predict compressed embeddings of merged tokens; RL stage | Llama-1B | fine-tune | −53% chain length for −4.8% accuracy vs CoT | [yes](https://github.com/xiaomi-research/colar) |
| [LOTUS](https://arxiv.org/html/2606.31779v2) (2026) | looped TF + parallel latent blocks supervised against CoT tokens | Llama-3.2-1B/3B | fine-tune | 3B GSM8K 70.0 vs CoT 71.5; 2.5× faster thought phase | [yes](https://github.com/yingfan-bot/lotus) |
| [Soft Thinking](https://arxiv.org/abs/2505.15778) | feed probability-weighted embedding mixtures instead of sampled tokens | ≤32B (R1-distill Qwen) | training-free | up to +2.5 pp pass@1, −22% tokens on math/code | [yes](https://github.com/eric-ai-lab/Soft-Thinking) |
| ↳ [critique](https://arxiv.org/abs/2508.03440) / [Soft Tokens Hard Truths](https://arxiv.org/abs/2509.19170) | — | ≤8B | — | models are "single-threaded": top token dominates the soft input. Continuous CoT trained with RL equals discrete CoT at pass@1 and beats it at pass@32; best recipe is train soft, infer hard | — |
| [Pause tokens](https://arxiv.org/abs/2310.02226) | learned filler tokens for extra width-wise compute | 1B | pretrain vs fine-tune | clear gains only when also used in pretraining; fine-tune-only is mixed | — |

Surveys: [latent reasoning survey](https://arxiv.org/abs/2507.06203) and [implicit reasoning survey](https://arxiv.org/abs/2509.02350).
Mechanistic analysis of Ouro, Huginn and retrofitted Llama: [2604.11791](https://arxiv.org/html/2604.11791v1).

## 3. What the literature says about retrofitting a *pretrained* model

1. **Loop the middle, keep the ends.** Every retrofit loops a contiguous mid-stack block and runs the first and last few layers
   once. The block sizes are ETD's 4 of 16, training-free's 3–4 of 28, and McLeish's 6–8. In retrofitted models the first and last
   "stages of inference" run once and the middle stages repeat (2604.11791). Training-free looping on Qwen3-1.7B-Base found n=4
   ([12–15]) best, with a sharp cliff at n≥6. For Qwen3-0.6B, n=3 was best.
2. **Naively re-applying layers without training hurts.** A loop run K times pushes the state to "t=K" of a residual ODE that the
   coda was never trained to receive (2605.23872). The fixes are to (a) damp the update, `x ← x + (1/K)(g(x) − x)` or a gate,
   (b) train, or (c) do both. Identity-preserving initialization, where K=1 reproduces the base model exactly, is what makes a
   low-budget retrofit safe (Shapiro 2026, LoopUS gate).
3. **Input injection stabilizes.** Re-adding the prelude output or embedding each iteration makes the state converge to fixed points
   quickly. Without it, the state drifts and degrades when you extrapolate to more loops (2604.11791). McLeish uses concat → linear
   2h→h. Huginn needed sandwich norm and a low learning rate, and its failed runs collapsed so that every token had the same hidden
   state.
4. **The token budgets are large.** McLeish used 26–50B tokens, ETD 50B, LoopUS 3B (billed as "20× fewer" than earlier work), and
   Relaxed Recursive multi-B. **No published retrofit uses fewer than ~1B tokens.** A 15–20 min GB10 run gives 1–4M tokens, which is
   **three orders of magnitude less**. Nothing in the literature predicts what happens at that scale.
5. **Iteration count.** Gains saturate at 2–4 loops in retrofits (LoopUS 3–4, ETD MATH k=3, LoopCoder 2, Ouro was cut from 8 to 4
   for stability). Random-depth training (Huginn, McLeish) lets you raise K at test time, but extrapolation is only about 1.5× the
   trained depth.
6. **Typical gains.** On math, +5–12 pp GSM8K after tens of billions of tokens. On general benchmarks, +0.5–2.6 pp training-free or
   +2 pp after 3B tokens. Knowledge-heavy tasks like MMLU barely move. Looping helps multi-hop *manipulation*, not storage (Ouro).
   **Code and agentic results so far come only from from-scratch models** (Ouro, LoopCoder-v2) or from a weak retrofit (tool-calling
   paper). No paper reports a retrofitted Qwen gaining on code.
7. **Continuous-thought tokens on pretrained models.** They consistently land at or below CoT-SFT at ≥1B (Coconut, CODI, LOTUS 70.0
   vs 71.5). They are expensive to train because each latent is a sequential forward pass and gradients go through n steps. They
   are also fragile, since latents homogenize and collapse. Coconut only beats CoT on search-like synthetic tasks (ProsQA). For "a
   better model per token" they are a weaker bet than depth loops. For "shorter reasoning traces" they are the right tool.
8. **Engineering.**
   - **KV cache.** Each loop iteration of an attention layer needs its own K/V, or K/V shared from the last iteration (Ouro: "last
     step only" costs 78.85 vs 78.92 GSM8K; "first step only" collapses to 18.7). In HF, give each iteration a virtual `layer_idx`.
   - **Serving.** vLLM needs a custom model class; SkyRL has an open, unmerged
     [looped-LoRA Qwen3 PR](https://github.com/NovaSky-AI/SkyRL/pull/2286).
   - **Qwen3.5 (hybrid).** Every 4th layer is full attention (`lllF` ×6 in the 24-layer 2B, ×8 in the 32-layer 4B). The Gated
     DeltaNet layers carry a per-sequence recurrent state, so each loop iteration needs its own state, and the fla kernels on sm_121
     are a further risk. **Start on dense Qwen3.**

## 4. Proposed experiment: `looped-latent` v0

**Question.** On the same tiny LoRA budget, does an identity-initialized mid-block loop (K>1) beat the same model fine-tuned
without loops, on a held-out task that needs serial computation per token?

**Base model.** **Qwen3-0.6B** (cached; dense, 28 layers, d=1024). A 1.7B would get only about 1.4M training tokens in 10 min;
the 0.6B gets about 4M and leaves room for 3 seeds. Confirm on **Qwen3-1.7B** (cached) only after 0.6B passes. Both cached
checkpoints are the post-trained hybrid-think models. Use `/no_think` and answer-only formatting. Qwen3.5 and 30B-A3B are out of
scope for v0.

**Mechanism v0 (the simplest credible retrofit).**
- **Split.** Prelude L0–11, loop block L12–15 (4 layers; training-free found n=3, [13–15], best for 0.6B), coda L16–27.
- **Iteration:**
  `h ← h + a_j ⊙ (g(h + W_inj·p) − h)`, where
  - p is the prelude output,
  - `W_inj` is zero-initialized,
  - `a_j` is a learned per-iteration scalar or vector gate initialized so the K-loop output equals one pass
    (RK/anchor form of 2605.23872 with β=1).
- **At init, K=1 is bit-identical to the base model** (unit test: max logit diff < 1e-3).
- **Adapters.** LoRA r=16 on all 28 layers' attention and MLP, for **both arms**. The loop arm may add per-iteration LoRA on the
  loop block (relaxed recursion). Extra parameters ≤ 2% of the base.
- **Training.** Random K∈{1..4} per step (Huginn/McLeish style), full backprop (cheap at K≤4), with optional deep supervision of
  intermediate iterations through the coda (LoopUS).
- **Eval.** At fixed K_eval=4, plus a reported sweep over K=1..8.

**Data (frozen harness, no license issues).**
- **Primary: an in-house generator of straight-line Python "program tracing" problems.** Each problem has 2–12 dependent
  assignments, arithmetic mod 100, swaps, small `if`s and `for range(3)` loops. The prompt ends in `print(x)`, the answer is the
  printed value, and there is **no CoT**: the model must answer in ≤8 tokens.
  - **Why this task.** It is code-flavored and needs serial state tracking, which is exactly what loops provably help (Saunshi,
    Giannou). Supply is unlimited, and exact match is cheap: the eval is prefill plus a few tokens with no vLLM needed.
  - **Splits.** Train on depth 2–8. Held-out sets come from disjoint generator seeds with hash-dedup: ID depth 2–8 (2,000 items)
    and OOD depth 9–12 (1,000 items).
- **Secondary, reported but not gating:**
  - **CRUXEval-O** ([MIT](https://huggingface.co/datasets/cruxeval-org/cruxeval), 800 items, direct output prediction), a real-code
    transfer check that needs a download in PREPARE.
  - **GSM8K test answer-only** (MIT; already cached), as a math transfer check.
- **Retention guard.** Mean NLL on a fixed 300-sample slice of OASST2 (cached) must not rise more than 2% over the base model.

**Budget per experiment (~18 GPU-min).** About 11 min of training per arm and 3–4 min of eval. The arithmetic behind this: 0.6B at
~5 GFLOP/token with the loop, at ~30–35 TFLOPs effective, gives ~4M tokens (~25k problems). Memory stays under ~25 GB, which
leaves room for the owner.

**Baselines (fixed by the harness; the agent cannot edit them).**
- **B0:** the base model zero-shot. This is the calibration check: a well-sized task puts it at 5–40% ID.
- **B1 (primary, train-compute-matched):** same LoRA, data stream, optimizer and **wall-clock**, K=1 with no loop code. B1 sees
  ~1.4× more examples.
- **B2 (optional, inference-compute-matched):** B1 plus m learned pause tokens, where m is chosen so its inference FLOPs match K=4.
  This checks whether "more compute" alone explains any gain.

**Metric and decision rule.**
- **Primary metric:** exact-match accuracy on held-out ID, at K_eval=4.
- **Pass condition:** loop − B1 ≥ **+3.0 pp** mean over **3 seeds** (per-run SE ≈ 1.1 pp at n=2000), all three seeds positive,
  **OOD not worse than B1**, and the retention guard holds.
- **Always report:**
  - the accuracy-vs-K curve and the accuracy-vs-depth curve;
  - inference FLOPs relative to base.

**What the Codex agent may change (the surface is `loop.py` plus its config):**
- the loop window (start and size ≤ 6),
- K_train distribution and K_eval ≤ 8,
- the update rule (naive, damped Euler, gate, RK),
- input injection (none, add, concat),
- LoRA rank, targets and per-iteration LoRA,
- deep supervision and loss weights,
- the learning rate and schedule,
- truncated backprop,
- an adaptive halting head.

**What is frozen:** the generator and splits, the eval harness and answer extraction, the base model, the wall-clock budget, the
baseline arms, the parameter cap, the retention guard, and the metric.

**Guards.**
- **Identity test:** K=1 at init equals the base model.
- **Causal-mask and KV test:** looped decode equals looped full-forward on 20 items.
- **Split hygiene:** no train/eval hash overlap.
- **Budget:** training stops when the harness timer expires (the harness owns the timer, not the surface).
- **Output format:** answers are capped at 8 new tokens with no CoT.
- **Memory:** GPU memory ≤ 40 GB.
- **Inference cost:** inference FLOPs ≤ 2× base (LoopCoder and Ouro saw no gain past 2–4 loops).
- **Stability:** divergence (NaN, or loss above 2× B1) fails the run.

**PREPARE-phase checks, before any agent iterations:**
1. **Calibrate generator difficulty** so that B1 lands at 40–70% ID. Otherwise there is no headroom, or the task is too hard.
2. **Validate that the task needs depth,** using the locally cached **Ouro-2.6B** (T=1 vs T=4) and **huginn-0125** (r=4 vs 32).
   If native looped models show no gain from more loops on this task, it is the wrong task.
3. **Run B1 over 3 seeds** to measure the real seed noise, then freeze the minimum effect size (≥ 2.5× the seed SD).

## 5. Honest expectations and biggest risks

- **Plausible effect.** On the synthetic tracing task, somewhere from 0 to +10 pp over B1. The deepest ID and OOD buckets are where a
  loop can plausibly matter. On CRUXEval-O and GSM8K answer-only, expect about 0 to +2 pp, which is inside the noise at n≤1319. I put
  the prior that v0 clears +3 pp over the compute-matched B1 at **roughly 30–40%**. The published retrofit wins all used ≥1,000×
  more tokens. The training-free wins are <1 pp on average.
- **Most likely failure: no gain over B1.** The LoRA-only baseline learns the task as fast as the loop arm does. The damped loop then
  stays near identity (gates never open), so loops add FLOPs but do no work. **Check `a_j` magnitudes.**
- **Instability and collapse.** Hidden-state blow-up or homogenization (Huginn's failed runs, Ouro at 8 loops), and oscillation
  beyond the trained K (LoopCoder at ≥3 loops).
- **Goodhart on a synthetic task.** A win on program tracing says little about agentic coding. The owner's real target, a looped
  agentic coder, needs a vLLM model class (see the SkyRL PR), long-context KV handling and multi-B-token training. That is well
  outside a 20-min loop. If v0 passes, the next steps are (a) 1.7B, (b) CRUXEval-O as the gate, then (c) a longer detached
  "healing" run on open code data. The healing run is a separate idea, not an arlab inner loop.
- **Checkpoint mismatch.** The cached Qwen3 checkpoints are post-trained. The training-free study saw its biggest gains on *base*
  checkpoints and occasional losses on small distilled ones. If v0 is flat, try Qwen3-0.6B-Base (a download in PREPARE).
- **Continuous-thought tokens (Coconut/CODI) are a deliberate non-goal for v0.** They need CoT traces and n sequential passes per
  example. At ≥1B they have not beaten CoT-SFT. Keep them as a v1 arm only if depth loops show signal.
