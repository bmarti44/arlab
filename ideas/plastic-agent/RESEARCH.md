# Plastic agent: writing experience into weights in an unfamiliar environment (research brief)

Date: 2026-09-27.

**Scope.** An agent adapts its **parameters** (full weights, a LoRA, or a trained KV prefix) online, from its own interaction with
a new environment. It then does better there *without* the transcript in context.

**Related work in this repo:**
- Architectural memory (Titans, memory layers, RMT, SSM state, Infini-attention) is surveyed in
  `ideas/memory-architecture/RESEARCH.md`. It appears here only where it bears on adaptation.
- The single-context TTT experiment (memory-architecture Experiment 1: qTTT, context-as-weights on Qwen3-1.7B) has been merged
  into this track. It is being folded into `ideas/plastic-agent/TTT-DEEP-DIVE.md`. §7 below is the *environment-exploration →
  weights* experiment only.

**Method notes:**
- "(PDF)" marks numbers checked against the paper text.
- OpenReview served a bot challenge and Reddit was blocked to the fetcher, so no reviews or Reddit threads are quoted.
- **Opinion** marks opinion.

## 1. Bottom line

1. **Naive "fine-tune on what you saw" fails:**
   - Next-token training on a passage moves no-context SQuAD only from 32.7 to 33.5 ([SEAL](https://arxiv.org/pdf/2506.10943), PDF).
   - RAG beats unsupervised fine-tuning for knowledge injection ([Ovadia et al.](https://arxiv.org/abs/2312.05934)).
   - Fine-tuning on API-update docs does not teach use of the updated API ([CodeUpdateArena](https://arxiv.org/abs/2407.06249)).
2. **What works: distil the model-with-context into the model-without-context**, on synthetic queries the model writes itself.
   - [Cartridges](https://arxiv.org/abs/2506.06266) (self-study into a trained KV prefix) match ICL with 38.6x less memory and
     26.4x the throughput (PDF).
   - [SDFT](https://arxiv.org/html/2601.19897v1) scores 89% on new facts against 91% for oracle RAG.
   - Related variants: [Knowledge Modules](https://arxiv.org/abs/2503.08727), [prompt baking](https://arxiv.org/abs/2409.13697),
     [SIEVE](https://arxiv.org/abs/2604.02339), and the original [context distillation](https://arxiv.org/abs/2209.15189).
   - Why it works: ICL generalises more flexibly than fine-tuning, and training on the model's in-context inferences closes the
     gap ([Lampinen et al.](https://arxiv.org/abs/2505.00661)).
3. **Parametric beats in-context only in a few measured regimes:**
   - Structurally novel tasks: ARC TTT is 6x over the fine-tuned baseline, and BBH 10-shot goes from 50.5 to 57.8
     ([Akyürek et al.](https://arxiv.org/abs/2411.07279)).
   - Long-trajectory drift: [aTTT](https://arxiv.org/abs/2607.03441) (PDF) scores 54.3 when its summary text is put in weights.
     The same text in context scores 49.3, which is *below* no adaptation (51.4).
   - Context beyond the usable window: Cartridges stretch MTOB from 128k to 484k tokens.
   - Serving cost.

   Otherwise it at best *matches* ICL.
4. **Small models are the weak point.** At 3B, SDFT's self-teacher "is too weak… performance lags behind standard SFT". Gains
   start at 7B (+4) and grow at 14B (+7) ([SDFT](https://arxiv.org/html/2601.19897v1)). The usable local models are 0.6–4B.
5. **Forgetting is real, but manageable with per-environment swappable adapters:**
   - After learning new facts, NaturalQuestions F1 drops 89% with full FT, 71% with LoRA and 11% with sparse memory-layer FT
     ([Lin et al.](https://arxiv.org/abs/2510.15103)).
   - A LoRA cartridge cuts MMLU from 54.7 to 45.3; a KV-prefix cartridge only to 54.3 (Cartridges, PDF).
   - On-policy RL forgets less than SFT ([RL's Razor](https://arxiv.org/abs/2509.04259)).

## 2. Ranked approaches

Rank = expected value for an arlab experiment (0.6–4B models, ~15–20 GPU-min): evidence × feasibility × fit.

| # | Approach | Mechanism | Where knowledge lives | Evidence (headline) | Code | Forum sentiment / replication |
|---|---|---|---|---|---|---|
| 1 | **Self-study / context distillation**: Cartridges, SDFT, KM-DCD, prompt baking, SIEVE | model writes queries about the context; student (no context) matches teacher (same model *with* context) by KL on logits | KV prefix (Cartridges); LoRA (KM, SIEVE); weights (SDFT) | ≈ ICL at 38.6x less memory; prefix beats memory-matched LoRA by 4.5 ChrF on MTOB; SDFT ≈ oracle RAG at 7B, loses to SFT at 3B | [cartridges](https://github.com/HazyResearch/cartridges) | Positive. Extended by [Cartridges at Scale](https://arxiv.org/abs/2606.04557) (within 2–6 pts of ICL). Composition and rebuilds are costly ([2608.30647](https://arxiv.org/abs/2608.30647)). Self-distillation collapse and forgetting in continual use ([2607.01763](https://arxiv.org/abs/2607.01763), [review](https://arxiv.org/abs/2608.25936)) |
| 2 | **TTT on the task or episode**: ARC TTT, aTTT, TT-SI, TTT-NN | a few LoRA steps on demos, trajectory text or self-generated variants | per-task or per-episode LoRA | ARC 53% (8B), 61.9% ensembled; aTTT +5.0 ALFWorld, +4.9 SWE-bench Lite at 1.9x wall-clock; [TT-SI](https://arxiv.org/abs/2510.07841) +5.5 avg with 68x fewer samples; [TTT-NN](https://arxiv.org/abs/2305.18466) | aTTT uses vLLM runtime LoRA | **ARC TTT replicated**: ARChitects (53.5%) and MindsAI used it ([ARC Prize report](https://arxiv.org/abs/2412.04604)). aTTT authors: "mainly preserves existing competence rather than teaching new abilities" |
| 3 | **Test-time / online RL**: TTRL, TTC-RL, Cursor Tab | RL on unlabeled inputs (majority-vote reward), self-curated curricula, or live accept/reject signals | weights or LoRA; rank-1 LoRA matches full FT for RL ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)) | [TTRL](https://arxiv.org/abs/2504.16084) +159% AIME24 pass@1 on Qwen2.5-Math-7B; [TTC-RL](https://arxiv.org/abs/2510.04786) ~1.8x AIME25 on Qwen3-8B; [Cursor](https://cursor.com/blog/tab-rl) +28% accept rate from a 1.5–2 h deploy-train loop | [TTRL](https://github.com/PRIME-RL/TTRL), [ttc](https://github.com/jonhue/ttc) | Random or incorrect rewards also lift Qwen2.5-Math by 21–24 pts ([Spurious Rewards](https://arxiv.org/abs/2506.10947)), so TTRL-on-Qwen is partly elicitation (my inference). Needs a reward, which a simulator provides |
| 4 | **Self-authored update data**: SEAL, SCoL | model writes its own finetuning data and hyperparameters; outer RL rewards edits that help after the update | LoRA / weights | SQuAD 32.7 → 47.0 (vs 46.3 with GPT-4.1 data); ARC subset 72.5% vs 20%; 30–45 s per candidate edit (PDF) | [SEAL](https://github.com/Continual-Intelligence/SEAL) | Forgets over sequential edits (own §5). [Edit-strategy search](https://arxiv.org/abs/2601.14532) does not beat the human "Rewrite" template. [SCoL](https://arxiv.org/abs/2605.07076) learns which layers to update. HN: "what is new here, exactly?" |
| 5 | **Hypernetworks**: Generative Adapter, Text-to-LoRA, Doc-to-LoRA, LatentSkill | meta-trained net maps context to LoRA weights in one forward pass | generated LoRA | [GA](https://arxiv.org/abs/2411.05877): 66.8 single-document, but 28.0 on 200-doc CPT, below base (SEAL App. B.8). [D2L](https://arxiv.org/abs/2602.15902): near-perfect NIAH to 32k. [LatentSkill](https://arxiv.org/abs/2606.06087): +13.4 ALFWorld-unseen with 64% fewer prefill tokens than text skills | [doc-to-lora](https://github.com/SakanaAI/doc-to-lora), [T2L](https://arxiv.org/abs/2506.06105) | Amortises #1, but meta-training does not fit in 20 min. v3 target |
| 6 | **Offline specialisation from own exploration**: Early Experience, SERA | reward-free rollouts; train on next-observation prediction, reflections, or soft-verified synthetic PRs | weights / LoRA | [Early Experience](https://arxiv.org/abs/2510.08558): +2.3 to +18.4 across 8 envs. [SERA](https://arxiv.org/abs/2601.20789) (PDF): 32B on 8k synthetic Django trajectories reaches 52.2% vs 51.2% for its teacher | [SERA](https://github.com/allenai/SERA) | "Implicit world modelling" is the cleanest reward-free objective. SERA's margin is about 1 SE, on repos seen in pretraining |
| 7 | **Constrained / sparse updates**: sparse memory FT, O-LoRA, null-space | restrict where an update can land | memory slots; orthogonal LoRA subspaces | Sparse memory FT loses 11% NQ vs 71% for LoRA; [O-LoRA](https://arxiv.org/abs/2310.14152) | — | Memory layers need a different base model (see the memory brief). An O-LoRA-style constraint fits a surface |
| 8 | **Architectural fast weights / TTT layers**: TTT-Linear, TTT-E2E, FWP | hidden state *is* a model, trained per token; meta-learned init | in-architecture | [TTT-E2E](https://arxiv.org/abs/2512.23675): 3B scales with context like full attention, 2.7x faster at 128k. See also [TTT layers](https://arxiv.org/abs/2407.04620) and [Schlag et al.](https://arxiv.org/abs/2102.11174) | [e2e](https://github.com/test-time-training/e2e) | Needs pretraining; covered by the memory brief |
| 9 | **Knowledge editing**: ROME, MEMIT, AlphaEdit | closed-form rank-one or null-space MLP edits | a few MLP matrices | Sequential edits cause gradual, then catastrophic, forgetting ([Gupta et al.](https://arxiv.org/abs/2401.07453)) | [AlphaEdit](https://github.com/jianghoucheng/alphaedit) | [AlphaEdit repro](https://arxiv.org/abs/2606.26783): reproduced on its original models; does not generalise to new ones; degrades tasks and refusals at scale. Edits facts, not tool semantics |
| 10 | **Meta-learning**: MAML, learned optimizers, agent meta-RL | learn an init, optimizer or policy that adapts fast | init / in-context policy | [MAML](https://arxiv.org/abs/1703.03400), [VeLO](https://arxiv.org/abs/2211.09760). [LaMer](https://arxiv.org/abs/2512.16848) meta-RL: +11/14/19 on Sokoban / MineSweeper / WebShop, adapting *in context* | — | The best agent meta-learning results use no gradients at test time |
| 11 | **Sleep-time compute / "sleep"** | think offline about the context before queries arrive | text in context ([Letta](https://arxiv.org/abs/2504.13171)); weights in [LMs Need Sleep](https://arxiv.org/abs/2606.03979) | ~5x less test-time compute at equal accuracy; parametric consolidation plus "dreaming" | [letta](https://github.com/letta-ai/sleep-time-compute) | Letta's version is not parametric. The framing fits the consolidation step (§4) |

## 3. Agents in unfamiliar environments: what has been measured

- **Test-time adaptation for agents** ([2511.04847](https://arxiv.org/abs/2511.04847)) has two parts:
  - A per-episode vector on the last hidden state (one gradient step per turn, 3% latency) fixes only format and syntax.
  - "Dynamics grounding" is persona-driven exploration that writes transition rules **into context**. It takes GPT-4.1 from 2%
    to 23% on WebArena multi-site.

  The large win was the in-context one.
- **aTTT** ([2607.03441](https://arxiv.org/abs/2607.03441), PDF; Qwen3.5 4B/9B/27B, Gemma-3-12B):
  - Setup: rank-8 LoRA, lr 5e-4, 2 steps every 5 agent steps.
  - Naive online updates are unstable because the agent retrains on its own repeats. Downweighting repeated n-grams fixes this.
  - It contains the only in-context vs weights ablation found: 49.3 in context vs 54.3 in weights.
- **Benchmarks:**
  - [CL-Bench](https://arxiv.org/abs/2606.05661) (6 stateful expert domains): naive ICL beats dedicated memory systems.
  - [AgentStream](https://arxiv.org/abs/2608.00155): gains are "gated by model capability"; no method dominates. It tests
    context, memory and skill methods only.
  - [Self-evolving agents forget](https://arxiv.org/abs/2605.09315): retained simple-task performance falls to 41.8%.
  - [CodeUpdateArena](https://arxiv.org/abs/2407.06249) is the nearest "new API" benchmark, and parametric updates fare poorly
    on it.
- **Gap.** No study compares an exploration transcript in context against the same transcript distilled into weights on a
  provably unseen environment with held-out tasks. aTTT is within-episode, 2511.04847 is in-context, and SERA has no in-context
  arm.

## 4. The driving analogy (short)

- **Humans are not zero-shot either.** Unfamiliar and foreign drivers are over-involved in some crash types
  ([J. Safety Res. 2007](https://www.sciencedirect.com/science/article/abs/pii/S0022437507000850);
  [Intini et al. 2019](https://journals.sagepub.com/doi/10.1177/0361198119851446)).
- **Why humans transfer:** a strong general model (perception, physics, social norms, compositional causal models:
  [Lake et al.](https://arxiv.org/abs/1604.00289)) combined with fast local learning. Complementary Learning Systems theory
  describes the mechanism: the hippocampus stores episodes quickly, and replay slowly consolidates them into cortex without
  overwriting it ([Kumaran, Hassabis, McClelland 2016](https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(16)30043-2)).
- **Self-driving now has the general half.** [Wayve](https://wayve.ai/thinking/ai-500-roadshow-500-cities/) drove in 500 cities,
  219 of them with no local data. It notes that behaviours from other markets "diminish as native data is collected and
  incorporated into training". So local adaptation is **offline fleet retraining**, not in-car plasticity (see also
  [Waymo](https://waymo.com/blog/2025/11/safe-routine-ready-autonomous-driving-in-new-cities/)).
- **The LLM mapping:**
  - pretraining is the general model;
  - the KV cache / ICL is the hippocampus: fast and flexible, but lost at session end, paid for in prefill on every call, and
    weaker as context grows;
  - **the missing piece is consolidation.** Cartridges, SDFT and "LMs Need Sleep" attempt it. Lampinen et al. explain why naive
    consolidation (training on the raw text) loses what ICL knew.

## 5. What the forums think

- **Opinion (Dwarkesh Patel):** continual learning is "a huge bottleneck". Text compaction "will be brittle". He puts 50/50 on
  human-like on-the-job learning by 2032 ([post](https://www.dwarkesh.com/p/timelines-june-2025)).
- **Opinion (Nathan Lambert):** it is a systems problem: "the path to continual learning is more context and more horsepower"
  ([post](https://www.interconnects.ai/p/contra-dwarkesh-on-continual-learning)).
- **Opinion (Karpathy):** agents "don't have continual learning", and fixing it is about a decade of work
  ([podcast](https://www.dwarkesh.com/p/andrej-karpathy)). Much human learning is more like "system prompt learning" than a
  weight change ([X](https://x.com/karpathy/status/1921368644069765486)).
- **Opinion (Sutton):** LLMs lack learning from experience and are a dead end without it
  ([podcast](https://www.dwarkesh.com/p/richard-sutton)).
- **Opinion (LessWrong survey, n=11):** several respondents favour context over weights. Hammond: "in-context learning is
  (almost) all you need". Paleka: "no finetuning, don't overcomplicate things". 4 of 6 expect transformative AI to be CL-based.
  All agree CL raises adversarial-finetuning risk ([post](https://www.lesswrong.com/posts/qZrbhoaEALFTmyidr/perspectives-on-continual-learning-survey-results-and)).
- **Opinion (LessWrong, "continual learning overhang"):** the top rebuttal says the bottleneck is *data showing learning from
  feedback*, not architecture ([post](https://www.lesswrong.com/posts/Lby4gMvKcLPoozHfg/are-we-in-a-continual-learning-overhang-1)).
- **Opinion (HN on SEAL):** "we have no idea how to do continual learning"; 30–45 s per reward is impractical; learning and
  inference are "entirely separate" ([thread](https://news.ycombinator.com/item?id=44271284)).
- **Measured, not opinion:** only CL-Bench and aTTT's ablation directly compare the routes, and both are small.

**Replication scorecard:**

| Result | Status |
|---|---|
| ARC TTT | Replicated by independent teams |
| TTRL | Widely re-run; attribution confounded on Qwen-Math |
| SEAL | Extended, not independently re-measured |
| Cartridges | Extended by follow-ups, with composition caveats |
| AlphaEdit | Partially replicated |
| SDFT | Contested in continual settings |
| Spurious rewards | Model-family specific |

## 6. The problems

- **Forgetting and interference.** See §1.5; SEAL also degrades over sequential edits. Mitigations: per-environment swappable
  adapters; KL-to-base or on-policy objectives ([on-policy distillation to restore behaviour](https://thinkingmachines.ai/blog/on-policy-distillation/));
  orthogonal or sparse updates; LoRA "learns less and forgets less" ([Biderman et al.](https://arxiv.org/abs/2405.09673)).
- **Stability.** Self-training loops amplify their own artifacts (aTTT drift, self-distillation collapse). New facts learned by
  fine-tuning encourage hallucination ([Gekhman et al.](https://arxiv.org/abs/2405.05904)).
- **Compute per update.** Paid once per environment; ICL instead pays prefill on every call.

  | Method | Cost |
  |---|---|
  | Cartridges | ~30 min on 8xH100 for an 8B model (PDF) |
  | SDFT | 2.5x FLOPs and 4x wall-clock of SFT |
  | SEAL | 30–45 s per candidate edit |
  | aTTT | 1.9x wall-clock |

- **Safety of self-modification.** A few fine-tuning examples can strip safety ([Qi et al.](https://arxiv.org/abs/2310.03693));
  narrow fine-tuning can produce broad misalignment ([Emergent Misalignment](https://arxiv.org/abs/2502.17424)); mass editing
  erodes refusals (AlphaEdit repro); and an agent that trains on what its environment shows it is open to poisoning.
  Mitigation: keep adapters discardable and gated by an eval, and never merge them into the base weights.
- **When parametric loses:** raw facts learned with next-token loss; small models with weak ICL teachers; corrections, which
  precomputed memory can ignore ([2608.30647](https://arxiv.org/abs/2608.30647)); reversal-style generalisation (Lampinen); and
  procedures: "executing instructions embedded in training remains less reliable than providing them in-context"
  ([Programming by Backprop](https://arxiv.org/abs/2506.18777)).

## 7. Proposed arlab experiment: `plastic-agent` (exploration → per-environment LoRA)

**Question.** After exploring a never-seen tool environment, can a small model consolidate that experience into a
per-environment LoRA? The target is to match or beat the exploration transcript *in context*, at a fraction of the inference
tokens.

(The single-context TTT variant is handled in TTT-DEEP-DIVE.md; this experiment is about experience gathered by acting.)

**Environment: "FauxOS"**, a procedurally generated fictional tool world. It is a deterministic pure-Python simulator with no LLM
inside.

- Each instance comes from a fresh seed: 20–30 tools with random pseudo-word names (`vorn_melt`, `kesh_tally`), random argument
  orders and units, **inverted or renamed semantics** (`purge` archives; `list` is newest-first and skips locked items), one
  conversion constant, three error codes with specific meanings, and a hidden state of 30–60 objects.
- The semantics are the "drive on the left": pretraining priors are wrong, not just missing.
- **Provably unseen:**
  - generated from arlab seeds after the model's release;
  - PREPARE gate: no-adaptation success ≤ 15% *and* ICL minus none ≥ 15 pts on the public split, or the pack refuses to
    calibrate.

**Exploration (PREPARE; frozen and cached).**
- The base model runs a fixed explorer prompt plus 30% random probes, for about 250 tool calls per instance. The result is a
  12–24k-token transcript of calls, observations and errors.
- Freezing it gives arms (b) and (d) *identical* experience and keeps exploration out of the RUN budget.
- Surface-driven exploration is v2.

**Tasks.** About 40 held-out goals per instance, e.g. "archive every ORE item heavier than 3 kesh, then report the tally
checksum".
- The model emits a tool-call program of at most 12 calls. The simulator executes it, and one retry with the error message is
  allowed.
- Success = exact final state or answer, scored by the frozen simulator.
- Splits: public = 4 instances × 40 tasks per seed; private = fresh instances; holdout = further instances, spent only at
  FINALIZE.

**Arms.** All use the same prompt: tool *names* only, no docs.

| Arm | What it is | Role |
|---|---|---|
| (a) | no adaptation | frozen reference |
| (b) | ICL: the whole transcript in the prompt | frozen reference; the baseline to match at lower cost |
| (c) | general-capability guard | validity check, below |
| (d) | the surface's adapter; the baseline surface is naive next-token LoRA on the transcript (expected ≈ (a), per SEAL and CodeUpdateArena) | what the campaign optimises |

**Guard (c).** The adapted model runs a fixed 300-item battery:
- GSM8K and ARC-Easy subsets, fetched in PREPARE;
- one *other* FauxOS instance, with its own transcript in context (the "can it still drive in San Jose" check).

A run is `valid: false` if it scores more than 2 pts below the base model.

**Surface.** `surface/adapt.py`: `adapt(transcript, tool_names, gen, train) -> lora_dir`.
- `gen` is the frozen budgeted vLLM client on the base model. It may use the transcript in context and returns top-k logprobs.
- `train` is a frozen PEFT loop. The surface supplies the data, the loss (next-token, KL to teacher logits, KL-to-base on generic
  prompts), rank and targets, lr, steps and replay.
- Codex searches the literature's recipe space: self-study questions (Cartridges), teacher-logit distillation (SDFT, KM),
  next-observation prediction (Early Experience), reflection or rule text (SEAL-style), repeat-n-gram down-weighting (aTTT),
  KL-to-base.

**Frozen:** base weights, generator, transcripts, tasks, the eval loop and prompt, the guard, the scorer, and the token and time
budget.

**Metric.** Mean task success over instances and seeds (maximise). **MES = 0.06** over the naive-next-token baseline surface.
- Also reported: (a), (b), the **ICL gap closure** (d−a)/(b−a), and prefill tokens per task.
- SE: with 160 task-paired items per seed and 3 seeds, SE ≈ 0.02–0.025. Meeting arlab's MES ≥ 2.5·SE may need 4–5 seeds or 6
  instances; calibration decides.

**Model.** **Qwen3-1.7B** (dense, mature vLLM LoRA path); fallback Qwen3-0.6B. Stretch: **Qwen3.5-4B**, a stronger ICL
teacher, but hybrid Gated-DeltaNet/attention: its vLLM LoRA path must pass a PREPARE smoke test, and a KV prefix would cover only
its attention layers. (Qwen3-4B and Qwen3-30B-A3B are only stubs locally.)

**Time budget** (GB10; estimates, to be confirmed in calibration). Per instance:

| Stage | Estimate |
|---|---|
| Self-study generation: ~300 prompts × 300 tokens, vLLM with prefix cache | ~1 min |
| Teacher top-k logprobs | ~1 min |
| LoRA training: ~150k tokens × 2 epochs, 1.7B bf16 | 2–3 min |

That makes 4 instances ≈ 16 min. Eval with LoRA hot-swap adds ~2 min and the guard ~1 min, for **≈19 min**. That is at the
ceiling; drop to 3 instances or to 0.6B if calibration overruns.

**Honest expectations:**
- If the environment is tuned right, ICL minus none should be 20–40 pts.
- The naive baseline should close under 20% of that gap.
- A good distillation surface plausibly closes **40–80%**, i.e. +8 to +20 pts over baseline, which clears the 6-pt MES.
- **Beating ICL at 1.7B is unlikely.** It is plausible only if transcripts are long enough (≥ 20k tokens) for 1.7B ICL to
  degrade, which is the regime where Cartridges and aTTT won.
- The saving of ~15–25k prefill tokens per call holds either way.

**Biggest risks:**
1. **Weak teacher.** 1.7B ICL may be too weak a teacher (SDFT failed at 3B). Mitigations: the PREPARE gate and the 4B stretch.
2. **Between-instance variance** leaves the pack underpowered. Calibrate before fixing seeds.
3. **Generator overfitting.** The surface may learn generator-family priors such as "semantics are often inverted". This is
   arguably the legitimate "general driving skill", but the verdict should also score holdout instances from a *perturbed*
   generator with new inversion types.
4. **Budget overrun.** Teacher logprobs over 20k-token contexts are the costly step.
5. **Scope creep.** Keep the evaluator single-shot with one retry.

**v2:** sequential A→B adaptation in one LoRA, then re-test A (true forgetting); surface-controlled exploration at a matched
token budget; a KV-prefix (Cartridge) arm through an HF eval path; a Doc-to-LoRA-style hypernetwork once per-instance
distillation works.
