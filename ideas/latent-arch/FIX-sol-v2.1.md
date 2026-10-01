**My recommendation: strengthen composition training and make `acc_depth` the primary, with ID accuracy as a guard.** No published result justifies promising σ≤0.02 for your existing composite within 660 seconds.

1. **Diagnosis.** This looks like seed-dependent circuit acquisition observed at a fixed deadline. Classical grokking requires prior training-set mastery, which these logs do not establish. Small-transformer studies show that circuit formation times vary across seeds; permutation-tracking research finds progressively longer compositions learned during training and initialization-dependent mechanisms. [Nanda et al.](https://arxiv.org/html/2301.05217v3), [Li et al.](https://arxiv.org/html/2503.02854v3)

   Your schedule compounds this: k≥3 appears at 40% progress, but LR decay begins at 50%. Plain token averaging also underweights final answers and lets the task’s weight change with k. Unmasked packing creates a concrete training/evaluation difference: training records can attend to earlier records; evaluation records stand alone. These are plausible contributors, not demonstrated causes.

   Crucially, the [pilot](/ctx/pilot-v2.txt) gives **σ_ID=0.188, σ_depth=0.009, σ_composite=0.095**. Depth is the least noisy component. Dropping it would remove the architecture question.

2. **One concrete configuration.** Freeze these conditions across architectures:

   - **660 seconds**, one scratch initialization per run; existing parameter cap.
   - Keep **48 text / 16 program rows**. Increase task influence through weighting, preserving text exposure.
   - Sample **uniform k=1..6 throughout training**. Dense prefixes already supply easier subtasks. Mixtures can induce an implicit curriculum, although the supporting composition theory uses idealized models. [Wang et al., §5.4](https://arxiv.org/html/2505.23683v1)
   - Use full-vocabulary  
     `L = mean(CE_text) + 0.125 mean(CE_final) + 0.125 mean_record(mean_prefix CE)`.
   - Isolate **56 complete records per program row**, ignore the remaining 16 tokens, reset positions and recurrent state per record. Prefer separate 18-token program forwards over a dense 1024-token attention mask.
   - Keep existing peak learning rates; hold them constant through **80%**, then linearly decay to zero over **20%**.
   - Primary: **equal-k mean accuracy over k7..10**, **MES=0.04**—four absolute depth points, stricter than the former eight-point depth-only equivalent. Evaluate **8,000 depth records**, 2,000/k. Retain the ID and text guards.
   - Screen on one seed; require two fresh confirmation seeds and three holdout seeds. Include candidate seed variance in the power calculation.

3. **Expected numbers.** Measured current baseline: ID **0.499**, depth/new primary **0.199**, depth σ **0.009**. For the proposed training, my **engineering estimates** are ID **0.75–0.95**, depth **0.20–0.25**, and depth σ **0.01–0.02 while depth remains near chance**. Validate these on five calibration seeds; successful generalization could increase variance. Conservatively allowing independent item noise, σ=0.02 and 8,000 depth items gives **2SE≈0.035**.

   Report equal-k log-probability as a secondary diagnostic. Smooth metrics can expose progress hidden by argmax, but cannot guarantee low seed variance; “max k mastered” adds another threshold. [Schaeffer et al.](https://papers.nips.cc/paper_files/paper/2023/file/adc98a266f45005c403b8311ca7e8bd7-Paper-Conference.pdf) Larger evaluation sets reduce sampling error only. Two training seeds reduce 0.09 to about 0.064; reaching 0.02 requires approximately **21 independent runs**.

4. **Fallback.** If every arm remains at depth chance, start a new configuration with **training k1..3, depth evaluation k4..7**, retaining five states, all 44 operators, and the protocol above. Recalibrate: a powered comparison at chance can establish a null, but cannot provide useful screening signal.