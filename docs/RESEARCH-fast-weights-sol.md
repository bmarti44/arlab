**I would close plastic-agent as a novelty-seeking campaign, retain its worlds and consolidation recipe as baselines, and investigate a narrower question: can a pretrained small agent revise actionable knowledge through bounded, causal weight updates while preserving everything that remains valid?**

“Experience can be consolidated into weights” is established. “Weights can change automatically during inference” is also established. The interesting uncertainty lies in **what those updates actually learn, how they handle contradictory evidence, and whether they transfer beyond the task families used to train the updating mechanism**.

This review covers work available through **2 October 2026**. Published numbers below are authors’ reported measurements, not reproductions on your GB10. Proposed budgets and expected outcomes are forecasts. I read the requested repository context and made no modifications.

**1. Plastic-agent: useful experiment, weak novelty claim**

Your reported v2b result—**49% success versus 28% with the transcript in context, accompanied by a 15 percentage point GSM8K loss**—is interesting as an engineering result. It does not establish a new learning mechanism, and the forgetting exceeds plastic-agent v2’s existing **6-point GSM8K guard**. I would not describe that candidate as a successful general adaptation method.

These are the closest precedents:

| Prior work | What it establishes | Consequence for plastic-agent |
|---|---|---|
| **SEAL: Self-Adapting Language Models**, 2025 | A model generates its own adaptation data/instructions, then undergoes LoRA training. Knowledge incorporation reaches **47.0%**, versus **32.7%** without incorporation and **33.5%** from training on raw passages. On a deliberately selected, small ARC subset, adaptation succeeds **72.5%**, versus **20%** without the learned self-edit policy. | Self-generated consolidation data and adaptation recipes are already established. SEAL still performs explicit training; it does not satisfy the owner’s desired deployment mechanism. [Paper](https://arxiv.org/html/2506.10943v2), [code](https://github.com/Continual-Intelligence/SEAL). |
| **Online Experiential Learning—OEL**, March 2026 | Extracts knowledge from agent trajectories and consolidates it through context distillation. On **Qwen3-1.7B/FrozenLake**, reported success is **7.3% without experience, 23.8% with self-experience in context, and 31.1% after consolidation**. | This directly establishes the broad claim on your model size: agent experience consolidated into weights can outperform experience in context. [Paper](https://arxiv.org/html/2603.16856). |
| **Experience Distillation**, July 2026 | Distills an experience-conditioned teacher into a student evaluated without that experience. On six TaleSuite tasks, normalized score rises **18.5 → 43.8**, versus **45.6** for ICL: **93.4% of the ICL improvement retained**. On 749 curated SWE tasks, pass@1 rises **5.3% → 51.4%**, versus **76.4%** for ICL and **8.0%** for trajectory SFT. | Consolidating actionable agent experience, rather than merely factual text, is already demonstrated. The curated tasks and proprietary base models limit direct comparability. [Paper](https://arxiv.org/html/2607.21051). |
| **Cartridges**, 2025 | Uses synthetic self-study and context distillation to build reusable compressed context, reporting **38.6× lower memory and 26.4× higher throughput**. It trains KV prefixes rather than backbone weights. | Neither self-study nor amortizing a context across later queries is new. It also supplies a stronger compression baseline than raw transcript next-token training. [Paper](https://arxiv.org/html/2506.06266), [code](https://github.com/HazyResearch/cartridges). |

OEL also supplies a relevant failure: on its Sokoban comparison, **raw-trajectory consolidation scores 7.8%, barely above 7.5% without experience**, whereas extracted-knowledge consolidation reaches **21.4%**. That is a published explanation for why beating your naive raw-transcript LoRA baseline is a low novelty bar. [OEL](https://arxiv.org/html/2603.16856).

Your repository further narrows the claim:

- Exploration is **scripted**, with 120 call/observation records, approximately 2.5–2.8k tokens.
- Evaluated tasks require **one tool call**. This measures learning tool semantics and argument conventions, not autonomous long-horizon planning.
- The earlier 80% winner fell to approximately **7%** under rewording and new operation types, while ICL remained around **56%**.
- V2’s disjoint wording families, reserved operations and counterfactual twins are valuable safeguards. They improve the experiment’s credibility; they do not make consolidation itself novel. [Repository design](/repo/ideas/plastic-agent/IDEA.md), [v1 audit](/repo/ideas/plastic-agent/AUDIT-v1.md).

**What remains worth doing?** A final frozen evaluation is worth completing if already budgeted and if a candidate passes forgetting guards. That would establish a reliable small-model result under your unusually careful controls. I would not launch another open-ended campaign optimizing the same consolidation claim.

A genuinely different continuation would investigate **causal semantic revision, transfer to withheld operation families, or the preparation cost required for forward-only adaptation**. Those need a changed hypothesis and evaluation.

---

**2. What “weights change during inference” actually includes**

There are three different requirements:

| Requirement | Methods that satisfy it |
|---|---|
| **No separate per-environment training job** | Hypernetwork-generated adapters, analytic fast-weight writes, embedded local gradient updates. |
| **No backward pass or optimizer at deployment** | Hebbian/delta updates, closed-form associative writes, hypernetworks. |
| **No preparatory training beyond the cached pretrained model** | A much smaller set: some direct test-time adaptation methods and nonparametric memories. Most capable learned writers require prior training. |

Calling a backward pass inside `forward()` does not eliminate training. Conversely, changing a recurrent matrix can constitute a genuine fast-weight update even when the pretrained checkpoint remains unchanged.

For example, a delta-rule memory can update as

\[
S_{t+1}=S_t+\beta_t(v_t-S_tk_t)k_t^\top.
\]

If a pretrained layer reads that state through

\[
W_{\mathrm{effective},t}=W_0+U S_t V^\top,
\]

the agent’s effective weights change after every observation. Deployment can implement this with ordinary matrix operations and no autograd. The difficult part is learning—or otherwise obtaining—useful keys, values and read/write interfaces.

**Fast weights, local optimization and neural memory**

| Family | Established result | What remains open or limits applicability |
|---|---|---|
| **Fast weights / linear attention** | Linear attention is an additive associative fast-weight programmer; the delta rule replaces simple accumulation with prediction-error correction. This connection predates current LLMs. [Schlag et al., ICML 2021](https://arxiv.org/abs/2102.11174), [code](https://github.com/ischlag/fast-weight-transformers). | Finite state causes interference. Storing an association does not automatically infer an operation’s semantics or support composition. A pretrained transformer also lacks a trained interface for an arbitrarily inserted memory. |
| **DeltaNet, Gated DeltaNet, DeltaProduct** | DeltaProduct uses several generalized Householder updates per token to improve state tracking. Its published \(S_5\) experiments use **four Householders**, train sequences of length **128**, and test length **512**. The setup uses **2 million training examples, 100 epochs, 12 heads and head dimension 32**. [Paper](https://arxiv.org/html/2502.10297v7). | These are expressive architectures trained to use their state, not evidence that a plug-in delta matrix will give Qwen new procedural abilities. The preparation budget is radically different from 660 seconds. |
| **Newer delta-rule follow-ups** | September 2026’s **ComplexKDA** expands transition expressivity through complex-valued/rotation-capable updates. [Paper](https://arxiv.org/abs/2609.24797), [code](https://github.com/OpenEuroLLM/ComplexKDA). | Expressivity results do not establish reliable learning from a short, ambiguous tool transcript. This remains primarily an architecture/pretraining direction. |
| **TTT layers** | Hidden state is itself a linear model or MLP, updated using a local self-supervised gradient objective. The original study scales to **1.3B parameters and 26B training tokens**. [Paper](https://arxiv.org/abs/2407.04620). | The projections and learning dynamics are trained jointly. A frozen-Qwen retrofit needs additional work. The authors explicitly describe their [PyTorch implementation](https://github.com/test-time-training/ttt-lm-pytorch) as naive and slow; optimized implementations are separate. |
| **Titans** | Learns nonlinear memory using gradient “surprise,” momentum and forgetting, combined with attention. At **760M parameters/30B tokens**, one reported downstream average is **52.51 for MAC**, versus **48.69 for Transformer++** and **49.69 for Gated DeltaNet**. [Paper](https://arxiv.org/abs/2501.00663). | The gain is real in the reported setup, but it is not universal dominance or a training-free retrofit. The widely used [Titans PyTorch repository](https://github.com/lucidrains/titans-pytorch) is a community implementation. |
| **ATLAS** | Optimizes memory over current and past tokens, adds higher-capacity features and stronger memory optimization. Reports accuracy above **80% at 10M-token BABILong context**, after benchmark fine-tuning. [Paper](https://arxiv.org/abs/2505.23735). | That is an absolute accuracy claim, not an 80-point improvement. Benchmark-specific fine-tuning and substantial pretraining matter. I did not verify an author-released reproduction of the complete system; [atlas-torch](https://github.com/danielquintas8/atlas-torch) is unofficial. |
| **MesaNet** | Maintains sufficient statistics and solves a regularized local least-squares problem. At **1B parameters/50B tokens**, global reasoning is **45.03**, versus **44.84 Gated DeltaNet / 45.54 Transformer**; recall is **41.79**, versus **39.54 / 52.27**. [Paper](https://arxiv.org/html/2506.05233). | “Locally optimal” means optimal for the chosen quadratic memory objective. It does not mean optimal task memory. Full attention still wins substantially on recall. |
| **In-forward gradient computation** | Transformers can implement gradient-like learning algorithms on suitable tasks and trained representations. [von Oswald et al.](https://proceedings.mlr.press/v202/von-oswald23a.html). | This is not proof that ordinary pretrained LLMs literally execute the same learning algorithm. An empirical study on pretrained LLaMA found inconsistent ICL/GD equivalence. [Shen et al.](https://arxiv.org/abs/2310.08540). |
| **Differentiable Hebbian / neuromodulated plasticity** | Offline backpropagation can train both fixed weights and plasticity coefficients; a learned neuromodulator controls when connections change during use. Demonstrated on image association, meta-learning, mazes and language-modeling LSTMs. [Differentiable plasticity, 2018](https://proceedings.mlr.press/v80/miconi18a.html), [Backpropamine, 2019](https://openreview.net/forum?id=r1lrAiA5Ym), [code](https://github.com/uber-research/backpropamine). | “The network learns how to change its own weights without deployment backpropagation” is already old. Reliable pretrained-LLM interfaces and transfer beyond the training task distribution remain harder. |
| **MIRAS / Nested Learning / HOPE** | Generalize memory design across update objectives, retention rules, optimizers and timescales. [MIRAS](https://arxiv.org/abs/2504.13173), [Nested Learning](https://arxiv.org/abs/2512.24695). | Useful conceptual frameworks, not evidence that lifelong learning and forgetting are solved, or that a ready-made module can be attached to your cached Qwen without preparation. |

Your latent-arch closure is therefore appropriate as **“could not test at this scale.”** It is not evidence against fast weights. The positive control failed, and the published DeltaProduct training regime was much larger. A new campaign should require a working pretrained-model positive control before searching. [Local closure](/repo/ideas/latent-arch/FIX-sol-v2.2.md).

**Hypernetworks and practical pretrained retrofits**

These are more relevant to your machine than replacing Qwen’s backbone.

| Method | Mechanism and measured result | Remaining gap |
|---|---|---|
| **GenerativeAdapter**, 2024/ICLR 2025 | A trained generator turns contextual activations into low-rank adapters on frozen Mistral/LLaMA. Its recurrent aggregation already supports streaming. On Mistral/SQuAD, reported score at 512-token context is **48.8 versus 45.4 ICL**, but at 8k it is **33.8 versus 42.5 ICL**. Preparation includes a roughly **500M-parameter generator**, 1B tokens and a reported **20 hours on eight H100s** before instruction tuning. [Paper](https://arxiv.org/abs/2411.05877). | Neither one-pass contextual adapters nor streaming adapter generation is new. Long-context information loss remains substantial. |
| **StreamAdapter**, 2024 | Combines within-chunk processing and recurrent cross-chunk state, absorbed into low-rank weights. Deployment adaptation is gradient-free. [Paper](https://arxiv.org/abs/2411.09289). | Another direct precedent for streamed context-to-weights. Preparation and out-of-distribution procedural transfer remain important distinctions. |
| **Text-to-LoRA**, 2025 | Emits adapters from task descriptions. In its nine-adapter reconstruction experiment, benchmark average is **73.3**, matching task-specific LoRAs, versus **55.8** base. That experiment indirectly sees the benchmark adapters during training. [Paper](https://arxiv.org/html/2506.06105v1), [code](https://github.com/SakanaAI/text-to-lora). | A description can select a learned skill; it cannot supply hidden environmental facts absent from that description. Its broader task-generalization experiments help, but do not establish novel tool-world acquisition. |
| **Doc-to-LoRA**, February 2026 | Context-conditioned hypernetwork emits LoRA without per-document training. Reports **82.5% of ICL performance on SQuAD**, not 82.5% absolute accuracy. Preparation: **309M hypernetwork, five days on eight H200s**. Adapters from 8k chunks are concatenated, so rank grows with chunk count. [Paper](https://arxiv.org/html/2602.15902v1), [code](https://github.com/SakanaAI/doc-to-lora). | Not a constant-state solution as published, and not a cheap ready-made Qwen-0.6B module. Its interference results show that freezing base parameters does not preserve behavior with an adapter active. |
| **δ-mem**, May 2026, revised September | **Extremely close prior:** a frozen Qwen/SmolLM backbone with gated delta-rule state producing dynamic low-rank attention corrections. On Qwen3-4B, overall average **46.79 → 51.66**; one variant improves MemoryAgentBench **29.54 → 37.84**. Rank 8; **4.87M trainable parameters**; one epoch on **2,219 QASPER samples**, trained on eight A800s with one reported training seed. It already evaluates selective forgetting. [Paper](https://arxiv.org/html/2605.12357v2), [code](https://github.com/declare-lab/delta-Mem). | “Streamed delta-rule LoRA on frozen Qwen” is already claimed. Open issues include causal action learning, robust revision, stronger seed evidence, and transfer beyond the writer’s training distribution. |
| **In-Place TTT**, April 2026 | Retrofitted local updates to pretrained MLP down-projections. Qwen3-4B-Base RULER at 128k improves **74.8 → 77.0**. The reported preparation includes **35B continuation-training tokens**. [Paper](https://arxiv.org/abs/2604.06169), [code](https://github.com/ByteDance-Seed/In-Place-TTT). | “Drop-in” describes architectural compatibility, not negligible preparation cost. Reproducing the published recipe is unsuitable here. |
| **TTT-NTP**, June 2026 | Uses next contextual hidden states as write targets; inference performs a closed-form regularized write. With **Qwen3-0.6B-Base and 0.2B preparation tokens**, RULER average improves **68.55 → 71.43**. [Paper](https://arxiv.org/html/2606.21803), [code](https://github.com/yancyou/TTT-NTP). | Particularly relevant scale, but still requires learned preparation. Retrieval gains do not establish learning actionable tool semantics. |
| **GradMem**, March 2026 | Updates a small set of memory vectors through test-time gradients while keeping the backbone fixed. On a synthetic 96-pair retrieval task, eight vectors achieve **32.6% with one write, 88.4% with five**, versus **12.9%** for forward-only RMT. [Paper](https://arxiv.org/html/2603.13875), [code](https://github.com/yurakuratov/gradmem). | The impressive result is a meta-trained synthetic task. The second-order training ablation collapses. Memory-vector updates also differ from changing Qwen’s layer weights. |
| **FAAST**, May 2026 | Closed-form associative learning from frozen representations and supervised examples. On GPT-2-XL/WikiText-103, perplexity **17.41 → 15.35**, versus **13.57 LoRA**; FAAST with an in-domain-trained readout reaches **13.23**. Reports **0.2 GPU-hours adaptation versus 3 for LoRA**. [Paper](https://arxiv.org/abs/2605.04651), [code](https://github.com/baoguangsheng/faast). | Strong evidence for analytic adaptation, but labels/readout preparation matter. Authors explicitly leave compositional reasoning and structured prediction open. |
| **CAMELoT**, 2024 | Adds a training-free associative memory to a frozen pretrained transformer. [Paper](https://arxiv.org/abs/2402.13449). | Valuable practical comparator, but chiefly an external activation/KV memory, rather than newly learned procedural weights. |

A September 2026 adjacent result, **CLAW**, generates adapters from recent transitions for model-based RL. It reinforces that fast context-to-adapter adaptation in changing environments is established outside language agents too. Its jointly trained world models differ materially from a retrofit to cached Qwen. [Paper](https://arxiv.org/abs/2609.12278).

**Online and continual learning**

Ordinary online parameter updates are also longstanding: **Dynamic Evaluation** updates language-model parameters on the observed sequence before predicting subsequent tokens. This is genuinely online, but uses gradients. [Krause et al., ICML 2018](https://proceedings.mlr.press/v80/krause18a.html).

More recent practical methods include:

- **qTTT:** targeted gradient updates to query projections, retaining the context/KV cache. Reports **+12.6 points LongBench-v2 and +14.1 ZeroScrolls** for Qwen3-4B over evaluated subsets. It addresses using retained context, rather than consolidating a disappearing stream. [Paper](https://arxiv.org/abs/2512.13898).
- **Self-guided/evidence-selected TTT:** selecting the adaptation text matters. One July 2026 study reports random-span adaptation reducing Qwen3-4B Thinking LongBench-v2 **40.4 → 38.9**, while oracle selection reaches **45.9**. [S-TTT](https://arxiv.org/abs/2607.09415).
- **Self-distillation for continual learning:** demonstration-conditioned, on-policy teachers can reduce forgetting. But a subsequent study finds stronger forgetting and even collapse under some continual post-training settings. On-policy data is not itself a forgetting guarantee. [SDFT](https://arxiv.org/abs/2601.19897), [failure study](https://arxiv.org/abs/2607.01763).

**The failures that should shape this lab’s experiments**

1. **Perplexity improvement is not durable recall.** TTT-E2E reports a 128k speed advantage, yet its passkey retrieval at 128k is **6% versus 99% for full attention**. It also reports approximately **3.4× training overhead at 8k**. [Paper](https://arxiv.org/html/2512.23675).

2. **Frozen checkpoints can still suffer active interference.** In Doc-to-LoRA’s unrelated-context experiment, SQuAD performance falls **0.201 → 0.096** with the generated adapter active. Resetting an adapter restores the checkpoint; it does not demonstrate preservation during use. [Interference appendix](https://arxiv.org/html/2602.15902v1).

3. **Naive writes can be disastrous.** TTT-NTP’s unwhitened inner-product write reduces LLaMA’s RULER average **55.80 → 12.61**, versus **59.70** with its regularized write. Conditioning and write scale are central, not implementation details. [Ablations](https://arxiv.org/html/2606.21803).

4. **Published mechanisms can be difficult to reproduce.** Hugging Face reports unsuccessful Infini-attention replication behavior under repeated compression. Mesa’s implementation has also had a documented, subsequently closed NaN/gradient-explosion issue. These are engineering cautions, not refutations of the architectures. [Infini-attention investigation](https://huggingface.co/blog/infini-attention), [Mesa issue](https://github.com/fla-org/flash-linear-attention/issues/1187).

5. **Overriding old knowledge is not a new discovery either.** The 2026 *Override Gap* study already investigates interference between generated LoRA and pretrained beliefs, including selective layer scaling. “Increase adapter strength to override prior knowledge” needs a more specific contribution. [Paper](https://arxiv.org/abs/2604.23750).

For your GB10, I would use **small FP32 fast states with ordinary PyTorch operations and the already verified SDPA path**. Architecture kernels in [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) are useful references, but published CUDA/Triton performance should not be assumed to transfer to ARM/sm_121. Your [hardware notes](/repo/docs/spark-notes.md) support the conservative implementation route.

---

**3. Two open questions worth the compute**

These are specific empirical questions not established by the papers reviewed. The underlying learning rules are existing ideas. A positive result would need to establish the stated boundary, not claim invention of fast weights.

| Question | Why it is open and potentially new | Feasible experiment and informative negative |
|---|---|---|
| **A. Can bounded, causal fast weights revise tool semantics from sparse contradictory observations, preserve unchanged tools, and transfer to withheld operation families?** | δ-mem already demonstrates online memory and selective forgetting; generative adapters already stream. What remains unestablished is this combination of **executed action correctness, unannounced semantic change, disjoint wording, reserved operations, matched twins and active forgetting guards**, on a small frozen language agent. | Retrofit a small writer/read interface to Qwen; no deployment gradients or transcript replay. Compare with matched δ-mem, ICL/retrieval and strong offline consolidation. A negative with working controls identifies an action-learning/revision boundary rather than another scratch-training failure. **Approximately 3–4 GPU-days.** |
| **B. Can an untouched pretrained model acquire actionable world knowledge through analytic writes without episodic meta-training?** | TTT-NTP needs learned preparation; FAAST uses supervised associations and trained readouts; δ-mem trains a writer. The unsettled question is whether native Qwen representations supply a sufficient write/read geometry for **raw tool observations**, without task-trained projections or a library of known operators. | Test a small, fixed family of normalized delta/ridge writes using native activations and observed-token targets. Compare against the same interface after limited meta-training, plus ICL and offline LoRA. A positive would remove a meaningful preparation requirement. A negative is informative if the trained-interface positive control works. **Approximately 1–2 GPU-days.** |

Both should distinguish:

- **Binding:** learning that pseudoword tool `dax` performs a familiar operation.
- **Parameter learning:** learning its argument order, unit conversion or constants.
- **Operation-family transfer:** learning behavior outside the writer’s training operation catalogue.

Success on the first two should not be advertised as learning a new algorithm. Qwen may already understand the underlying operations from pretraining.

I recommend **A** first. It has a clearer measurement surface and a stronger route to a well-powered result. B is closer to the owner’s strongest interpretation of “without training,” but has a greater risk of failing because an untrained memory interface cannot communicate with the backbone.

---

**4. Recommended next pack: causal revision of tool memory**

**Hypothesis**

> After one-time training of a small memory interface, a frozen Qwen agent can incorporate a stream of tool observations through forward-only weight updates. After an unannounced change to two tools, it can recover actionable semantics from six verified observations, preserve unchanged tool knowledge, and transfer to reserved operation families—substantially better than a matched existing fast-weight baseline.

This deliberately requires more than reproducing δ-mem on another dataset.

**Model and editable surface**

Start with **Qwen3-0.6B**, with a preregistered **Qwen3-1.7B fallback only if the initial capability controls fail**. Freeze the backbone throughout.

A manageable implementation is:

- Four selected layer locations with effective corrections \(U_\ell S_\ell V_\ell^\top\).
- \(S_\ell\) is **32 × 32 FP32**: four states occupy **16 KiB**.
- **64 KiB maximum total persistent per-world state**, including every auxiliary buffer.
- At most **5M trained interface parameters**; these are shared across worlds.
- A writer receives **one complete current call/observation record** and the existing state.
- Deployment uses matrix writes only: **no optimizer, backward pass, teacher queries or previous-record KV cache**.

The research agent may edit the writer, normalization, local retention/update policy, and bounded readout implementation. The backbone, state-accounting hooks, data generator, evaluator, controls and budgets remain frozen.

One-time interface training is explicitly allowed. This pack tests **absence of per-world training**, not absence of all preparatory training.

**Frozen task design**

Reuse FauxOS, but add a causal revision phase:

1. Stream the initial 120 exploration records.
2. Without announcing the change, alter **two tools’ semantics, argument conventions or unit mappings**.
3. Supply **six successful, identifying post-change observations—three per changed tool**. Keep unaffected tools’ observations matched across conditions.
4. Evaluate new one-call tasks separately on changed tools, unchanged tools, familiar operations and reserved operations.
5. Clone the final memory for every evaluation task. Evaluation goals and responses must never update another task’s memory.

Freeze diagnostic curves at **1, 3, 6 and 12 observations**; six is primary. Include unchanged/sham-change worlds and counterfactual twins.

Use disjoint training/development/test wording families and world seeds. Reserve operation families from **all interface-training examples, synthetic targets and development selection**. Audit actual exposure, not just file names.

Keep multi-call composition secondary. Enable it only if ICL and a positive control solve it before sealing. The existing single-call test already avoids the latent-arch trap.

Also include a longer-stream stress test. **A 2.8k-token transcript is not evidence about extreme context length.** Raw text may fit inside the proposed byte budget, so a fair retrieval baseline must be allowed to retain it.

**Baselines**

| Baseline | Purpose |
|---|---|
| Frozen Qwen, no experience | Establishes unadapted performance. |
| Trained interface with writes disabled | Removes generic skill gained during interface training. |
| Full-transcript ICL, with shared-prefix caching | Strong accuracy reference; do not charge repeated prefill that the existing harness already avoids. |
| Byte-matched transcript storage with retrieval/summary | Tests whether weights help beyond inexpensive retained evidence. |
| Matched δ-mem-style and additive fast-weight writers | Closest mechanistic precedents. Train on the same permitted worlds and count preparation compute. Label a port accurately rather than claiming to reproduce published numbers. |
| Strong offline consolidation | Best **guard-passing** plastic-agent/context-distillation recipe, tuned on training/development worlds, using the same observed prefix and a fixed **160-second per-world budget**. Include raw LoRA as a diagnostic, not the principal comparator. |
| One gradient update per observation | Separates “no separate job” from genuinely backward-free learning. |
| Wrong-world/twin state and reset state | Measures whether gains require the correct observations. |

For the offline baseline, allow verified train-only pseudoqueries or an experience-conditioned teacher. Comparing a learned writer only against raw next-token transcript LoRA would understate established consolidation methods.

**Positive controls before search**

Require all three:

1. **Evidence sufficiency:** full ICL beats no-experience performance by at least **15 points**, and reaches at least **25%** on each primary development stratum.
2. **Task solvability:** an explicit, readable oracle tool specification reaches at least **80%**.
3. **Interface capacity:** optimize the proposed fast state on **development-only training tasks**, then evaluate different development tasks. It must beat the disabled-write interface by **15 points across three seeds**.

The third is an oracle diagnostic, never an eligible method. It answers the question latent-arch left unresolved: **can this model and interface express a solution at all?**

If ordinary δ-mem already satisfies the entire intended capability claim, stop and report successful transfer of an existing method. Do not manufacture an architectural novelty claim.

**Primary metric and minimum effect**

Primary outcome: **world-macro-averaged exact executed task success after six post-change observations**, balanced across familiar and reserved operation strata.

Preregister:

- **MES: +6 percentage points** over the strongest matched existing forward-only memory baseline, chosen and locked using development data.
- Paired, world-clustered **95% confidence interval excludes zero**.
- **Noninferiority to strong offline consolidation: margin 5 points**, assessed with a confidence interval.
- Reserved-operation improvement must independently be positive; familiar-only gains support a narrower claim.
- Correct-world versus twin evidence must account for at least **half the improvement over the disabled-write control**.
- Report full-ICL accuracy and latency without asserting superiority unless measured.

Forgetting guards:

- **At most 2-point loss** on GSM8K and unrelated-world tasks **with the learned memory active**.
- **At most 2-point loss** on unchanged-tool tasks across the revision phase.
- Reset restores the original model’s behavior, as a separate engineering check.

Do not pool the forgetting metrics: improved tool performance must not conceal degraded math. Do not disable memory using a benchmark-specific classifier.

Aim for **64 sealed test worlds, three final training seeds**, with enough tasks per world to balance the strata. Calibrate power on separate worlds before sealing; enlarge the sample only before the campaign if world-level variance demands it. Three seeds on the same worlds are not three independent sets of worlds.

**Compute budget**

A reasonable hard cap is **96 GB10 GPU-hours**:

| Work | Cap |
|---|---:|
| Throughput calibration and positive controls | 8 h |
| Six candidate configurations | 24 h |
| Matched baseline training | 12 h |
| Two additional seeds for the selected candidate | 8 h |
| Offline adaptation baselines and frozen evaluation | 32 h |
| Reserve for measured overhead | 12 h |

Suggested starting data: **512 training worlds and 20–40k verified training tasks**, with cached frozen encoder features where valid. Limit each candidate by both wall clock and processed tokens; caching must not create uncounted training exposure.

These are scheduling estimates. If measured throughput cannot support the powered evaluation within the cap, stop or reduce scope **before sealing**, rather than shrinking the hidden test afterward.

Use FP32 memory updates and ordinary PyTorch first. A custom-kernel project would consume the research budget and confound the scientific question.

**Ways it could fail or be gamed**

- **Unidentifiable observations:** several semantics fit the same examples. Oracle-spec performance alone cannot establish that the stream identifies the answer.
- **Generic tool training masquerading as adaptation:** catch with disabled writes, twins and withheld operations.
- **Hidden transcript retention:** count all persistent tensors, Python objects, caches and helper-process state.
- **Evaluation queries becoming training data:** freeze cloned memory during evaluation.
- **Exact-answer lookup:** retain v2’s exclusion of tasks answered by repeating logged calls.
- **Selective forgetting by indiscriminate reset:** changed-tool recovery must coexist with unchanged-tool retention.
- **Adapter-induced damage:** evaluate general abilities with memory active.
- **Weak baselines or unequal compute:** charge teacher generation and count interface preparation; preserve ICL prefix reuse.
- **Winning only on familiar operators:** report that limitation rather than calling it general environment learning.
- **Floor effects:** stop when positive controls fail.

**Preregistered stop rule**

- **By eight GPU-hours:** stop as *inconclusive* if capability, evidence or interface controls fail.
- **After six configurations or 24 search hours:** stop searching. Select only from guard-passing development candidates.
- Run **one sealed final evaluation**, with three predetermined training seeds.
- Declare *supported* only if the MES, confidence, offline noninferiority, reserved-operation, twin and forgetting conditions all pass.
- Declare *not found at this scale* if controls work and the upper confidence bound excludes the 6-point gain.
- Otherwise declare *inconclusive* or state the narrower supported result.
- Do not reopen the test split, change the MES, or add configurations after seeing the final result.

**My forecast:** preserving unaffected knowledge while revising a few tool mappings is a more plausible first gain than beating full-transcript ICL. Transfer to withheld operation families is the hardest part and could fail despite excellent associative recall.

That failure would still answer a useful question: whether a small pretrained fast-weight agent learns new actionable structure, or mainly compresses bindings within a repertoire its writer already knows. That is a substantially stronger research target than another campaign showing that transcript distillation can improve a familiar tool world.