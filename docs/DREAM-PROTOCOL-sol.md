Recommend a **paired online experiment on fast tasks, followed by a small nanochat transfer check**. Within 1–3 GPU-days, modest nanochat policy effects will usually remain inconclusive.

Dream-RSI motivates testing, but its monotonic guarantee concerns training replay; it reports no seed uncertainty or held-out policy trees. An independent toy follow-up found replay improvement followed by worse fresh quality. AIRA shows that operators can dominate policy choice and rankings change with budget. [Dream-RSI](https://arxiv.org/html/2609.14858v1), [independent follow-up](https://github.com/fspecii/dream-rsi-lab/blob/main/docs/experiments.md), [AIRA](https://arxiv.org/html/2507.02554v2).

1. **Claim and primary metric**

   Pre-register: “For this frozen dreamed policy, pack/root distribution and resource allowance, expected independently evaluated improvement exceeds **each** fixed baseline by >0.2 MES, at no greater cost.”

   Primary: **mean holdout d/MES of the program selected using screen/confirmation only**, with root fallback when no candidate qualifies. Never maximize over holdout scores. This measures final recommendation quality, consistent with simple-regret evaluation. [ICML 2023](https://proceedings.mlr.press/v202/zhao23g.html).

   Continuous terminal quality is my statistical-efficiency choice, subject to pilot variance. Other endpoints:

   - **AUC:** secondary; correlated checkpoints do not automatically improve power, and screen-best AUC rewards selection noise.
   - **Time to 1 MES:** useful operationally; retain failures as nonattainment within budget, rather than averaging successful runs. [COCO methodology](https://ieeexplore.ieee.org/abstract/document/9905722).
   - **Probability of ≥1 MES:** discards magnitude and already saturates in T2.
   - **V:** requires predeclared cost preferences; the paper also includes a parallelism bonus absent from arlab’s V.

   Register “similar quality, cheaper” separately as noninferiority plus savings.

2. **Replication, power and holdout noise**

   The unit is a **complete search round**, blocked by root, evaluation randomness and execution window. Nodes and holdout seeds are not independent policy trials. Dream revisions are not replicates. Independent train/dream cycles are necessary to generalize beyond one frozen policy. [Variance methodology](https://arxiv.org/abs/2103.03098).

   With a policy margin of 0.002 bpb, true advantage 0.004 and paired round SD \(s_D=0.004\), plan approximately **48 rounds/arm** for ≥80% joint power against both controls. At \(s_D=0.005\), approximately **72**. These are assumptions, not estimates from two fixed rounds. Full nanochat testing would cost hundreds of GPU-hours.

   T2’s “one holdout” already means one selected program evaluated on five seeds. Compare candidates directly on shared fresh seeds: the common root cancels. Estimate paired covariance rather than combining root-relative error bars. [stats.py](/repo/arlab/stats.py)’s \(\sqrt2\sigma/\sqrt n\) does not estimate between-round policy uncertainty. Preserve top-three confirmation; add evaluation seeds where measurement noise dominates.

3. **Controls and replay validity**

   Freeze proposer/version, effort, prompts, repair rules, harness, root and workers. Randomize/interleave arms; prohibit cross-round memory. Common random numbers help, but nominal LLM seeds do not guarantee identical outputs. Share a prescribed opening batch, redraw it per block, and charge it to every arm. Reuse later proposals only for identical complete inputs. [CRN research](https://arxiv.org/abs/1910.09259).

   Add **handwritten parallel refinement stopping at the first screen ≥1 MES**. Its observed prefixes require no extra proposals, although separate finalization costs trials. This isolates T2’s exercised behavior.

   Separate training trees, dream-gate trees and final online evidence. Gate one training-selected candidate once; repeated gate reuse makes it development data.

   Replay remains diagnostic. Unsupported branches and deterministic logging do not satisfy classical unbiased replay assumptions. [Bandit replay](https://arxiv.org/html/1003.5956v1). Moreover, [online.py](/repo/arlab/tree/online.py:217) exposes global best/recent attempts and sibling counts: changed traversal changes proposal context, while replay preserves old proposals.

   Forecast rankings on untouched broad trees; verify against independent online blocks using prediction errors, sign concordance and block-bootstrap intervals. Three aggregate policy points are insufficient validation.

4. **Concrete registration and cost**

   Use **ttc-controller’s cached CPU trials** for the powered stage; exclude the saturated fixture. State that results are conditional on this task/cache. Smaller training or search budgets change the estimand; other packs require transport evidence.

   Collect six broad 24-node development trees: four training, two gate; evaluate replay at the intended 12-node horizon. Dream \(M=8\), then freeze the policy or record rejection.

   Run **48 fresh blocks**, each containing dreamed, parallel refinement and greedy: W=4, 12 attempted nodes/arm, a shared four-node opening, identical confirmation, and five fresh paired holdout seeds. Predeclare resource caps including finalization. Charge failures; amortize development over 48 deployments. Track GPU-hours, Codex calls and priced tokens separately.

   Use paired t contrasts, **one-sided α=0.025/comparator**, with block-bootstrap sensitivity; no outcome-driven stopping.

   - **Proven:** both quality lower bounds exceed 0.2 MES and cost upper bounds show no increase.
   - **Denied:** either quality upper bound falls below 0.2 MES.
   - **Inconclusive:** otherwise.

   These verdicts concern the meaningful, scoped claim.

   Base usage: **144 development +8 dream +1,344 test calls**. Three fresh 32-node nanochat triplets add ≤264 calls with shared openings and approximately **22.5–36 GPU-hours**. Including T2b, budget roughly **1,900–2,000 calls before repairs**, at least four quota-days, and ≤56 GPU-hours plus cache preparation. At 24 GPU-hours, omit fresh GPU triplets. Three nanochat blocks cannot establish superiority there.

5. **T2b**

   Useful as a pilot of stricter eligibility, stopping frequency and variance. Its 1.05× timing guard does not match total search cost or FLOPs. Do not pool with T2: proposer and eligibility changed.

   c-a0/a1 train/gate the policy; add untouched parallel controls after freezing and another greedy round. Randomize subsequent triplets and retain rejected dreams in the record. “Underpowered=false” concerns candidate-versus-root effects, not policy comparisons.

   Measure GPU reservation and trial time separately. [report.py](/repo/arlab/tree/report.py:74)’s active hours include proposal latency and omit finalization.

Additional references:

- [SimpleTES](https://arxiv.org/html/2604.19341v2): budget allocation and task-dependent pruning.
- [AIDE](https://arxiv.org/html/2502.13138v1): fixed search baseline.
- [ML-Master](https://arxiv.org/html/2506.16499v1): search, memory and reasoning design; cross-paper hardware differs.
- [NeurIPS evaluation methodology](https://arxiv.org/abs/2108.13264): uncertainty across runs and tasks.
- [Dream-RSI critique](https://www.laura-martel.com/blog/dream-rsi-replay-simulator): aggregation, replay validity and omitted dreaming costs; commentary rather than peer-reviewed evidence.