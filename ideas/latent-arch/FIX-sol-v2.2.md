**Recommend (e): a bounded positive-control calibration before launching the campaign.** My previous advice mistook a low-variance floor for a useful screen; the ID forecast was also too optimistic. Nothing modified.

1. **Mechanism.** These results suggest learned one/two-step transitions and a partially acquired composition circuit, followed by an uninformative fallback. They do **not** identify whether that circuit is sequential, associative, or memorized. Transformers trained on permutation composition can learn hierarchical associative scans with seed-dependent emergence. [Li et al.](https://arxiv.org/html/2503.02854v3)

   First verify “exact uniform”: mean target log-probability ≈−ln(5) and identical accuracies do not prove uniformity on individual examples. Inspect 256 depth prompts: five-state probability vectors, entropy, total state probability mass, and predicted-class frequencies.

   If uniformity holds, it is a natural failure mode. Averaging over an unknown random derangement maps state probabilities toward uniformity, shrinking deviations by **−1/4 per unknown operator**. Four unknown transitions leave at most **0.003125** deviation from 0.20. Losing the incoming state therefore makes later known permutations useless; cross-entropy rewards the uniform fallback.

   v2.1 changed curriculum, loss normalization, record isolation, and LR simultaneously. Removing the easy-first curriculum plausibly impeded circuit acquisition; text/task interference and seed variation remain alternatives. More exposure cannot establish causality. Also, the logs show **38.97M versus 20.77M program input tokens: 1.88×**, not 4×. v2 seed 1 was the strongest of three.

2. **Validity and signal.** The pack is a valid operational comparison of **architecture plus training recipe under this hardware budget and guards**. It measures composition-depth extrapolation at fixed input length. It is presently **unvalidated as a useful search screen**: all tested arms fail before the evaluation range. Switching to logp cannot manufacture signal when predictions are uniform; “max k mastered” adds threshold noise.

   Nevertheless, ≥0.04 improvement is plausible with structured recurrence. **DeltaProduct with four Householder factors per token and negative transition eigenvalues** learns S₅ and extrapolates from 128 to 512 operators; single-factor DeltaNet fails to fit S₅ even with ten layers. Its experiments use much longer training, so this establishes possibility, not 660-second feasibility. [Siems et al.](https://arxiv.org/html/2502.10297v7)

   Looped transformers also extrapolate, but training matters: Fan et al. supervise computation budgets; a recent preprint reports successful extrapolation on fully bijective finite-state composition and failures depending on recurrence configuration. Merely repeating blocks is insufficient. [Fan et al.](https://arxiv.org/html/2409.15647), [Liang et al.](https://arxiv.org/html/2609.33144)

3. **Exact next step.** Retain v2.1’s task, losses, **660 seconds**, and **8,000 depth records**. Finish baseline seeds 3–5. Then run **three seeds, 1–3**, of one prospective positive control:

   - Baseline GPT plus a learned DeltaProduct residual branch: **one layer, four heads, 16-dimensional keys/values, four Householder factors/token, β∈[0,2]**.
   - Generic computation on every token, zero initial recurrent state, pure-PyTorch parallel affine scan; learned projection into the GPT residual stream.
   - New branch: AdamW **LR 0.001, betas (0.9, 0.999), weight decay 0.000001**, existing 80% constant-LR schedule; retain backbone recipe.

   This scaled branch is an engineering proposal, not a replicated result. Require **all guards**, each seed **acc_depth≥0.24**, mean **≥0.26**, and sample σ **≤0.02** before opening the campaign.

   Expect baseline **≈0.20**, σ near zero *while fallback persists*; independent-item SE is **0.0045**. Candidate σ is unknown. If the control fails, stop: useful signal within this budget remains unestablished.