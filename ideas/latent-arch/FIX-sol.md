**The most likely fix is a task redesign: ordered permutation composition with dense state supervision supplied only to the loss.** It preserves the architecture comparison, but its success within 330 seconds still needs a pilot.

**1. Root cause**

The strongest explanation is an optimization bottleneck combining variable lookup, arithmetic, and sparse execution supervision. Even `k=1` requires finding the queried assignment, resolving its randomly named operand to a constant assignment, then calculating the answer. Removing binary operations leaves those requirements intact. [Generator](/pack/frozen/prepare/progen.py:46)

Every program occupies 84 tokens including BOS and answer; only the answer directly supervises execution. The trainer uses shifted next-token targets, and the model averages cross-entropy over every token. Consequently, answers occupy approximately **0.11% of targets at 58/6 rows**, **0.29% at 48/16**, and **1.10% at 4/60**. Syntax learning can dominate the reported loss while execution remains at chance. [Serialization](/pack/frozen/prepare/progen.py:41), [trainer](/pack/frozen/run/trainer.py:61), [loss](/pack/surface/model.py:159)

This diagnosis has unusually close empirical support: Wu et al.’s approximately 38M-parameter transformer learns variable dereferencing **without arithmetic**, but exhibits prolonged heuristics and a systematic-binding transition around 14,000 updates—beyond this pack’s 7,300-step attempt. Induction-head research likewise documents abrupt circuit emergence, although it does not establish this pack’s precise mechanism. [Wu et al.](https://arxiv.org/html/2505.20896v1), [Olsson et al.](https://arxiv.org/abs/2209.11895)

Calling this *grokking* would be premature: the training-program probe also remains at chance, whereas classical grokking follows successful training-set fitting. Easy-to-hard composition research supports curriculum as an optimization intervention. [Pilot](/ctx/BLOCKED-B2.md), [Power et al.](https://arxiv.org/abs/2201.02177), [Wang et al.](https://arxiv.org/html/2505.23683v1)

I found **no source-level answer mismatch**: pieces are assigned distinct single-token IDs directly, and training and evaluation apply the same vocabulary permutation. RoPE is implemented; the closely related binding study also succeeds with RoPE. Positional changes alone have weaker support here. One concrete reproducibility defect: current code specifies **mod 10 and 48/16 rows**, contradicting the log’s claimed restoration to mod 100 and 58/6. [Preparation](/pack/frozen/prepare/prepare.py:110), [model](/pack/surface/model.py), [configuration](/pack/frozen/run/common.py:15), [pilot](/ctx/BLOCKED-B2.md)

**2. Recommended fix**

Replace the arithmetic programs with this explicitly proposed configuration:

- **Format:** `BOS INIT s0 t1 … t14 ?`, exactly **18 tokens**. Initial state `s0` is uniform over five states. Each active operator is one atomic token denoting one of the **44 permutations having no fixed points**; inactive slots contain identity `NOP`. Randomly place exactly `k` active operators among the 14 slots. The answer is their ordered composition applied to `s0`.
- **Splits:** retain training/ID `k=1..6`, OOD `7..10`, exploratory `11..12`. Reserve operator tokens before tokenizer training, retain vocabulary size 8192, assert distinct single-token encodings, and permute inputs and labels together.
- **Mix/curriculum:** **48 text / 16 program rows**. First 20% of wall-clock progress: `k=1`; next 20%: uniform `1..2`; remaining 60%: uniform `1..6`.
- **Supervision:** provide the state after each active operator as a sidecar target at that operator’s position, plus the final state at `?`. **Never insert intermediate states or answers into program inputs.** Ignore program syntax targets. Use full-vocabulary cross-entropy:
  `L = mean(CE_text) + 0.125 mean(CE_final) + 0.125 mean(CE_prefix)`,
  averaging prefix loss within each example first.
- **Packing:** fit 56 complete records per program row, mask padding, isolate records with causal segment masks, and reset recurrent state between records.

Prefix-state supervision without feeding states back is an established state-tracking objective; sparse supervision particularly harms transformers in automata experiments. Noncommuting permutations retain order-sensitive composition while removing variable binding and arithmetic. [Li et al.](https://arxiv.org/html/2503.02854v3), [Liu et al.](https://arxiv.org/html/2210.10749v2)

Put task generation, sidecars, segmentation, curriculum, and shared loss specification in the **frozen harness**; expose generic metadata to surfaces. Start a new data-hash campaign and recalibrate the baseline text guard, retaining scratch initialization, 330 seconds, and parameter/inference limits. These are common experimental conditions. [Existing design](/pack/IDEA.md)

**My forecast:** approximately **60% ID accuracy**, plausibly 50–70%, around 1,300 updates. This is an unvalidated engineering estimate. Rigorous evidence establishes that tiny transformers can learn index lookup within a few thousand steps and can learn densely supervised permutation composition; **I found no rigorous demonstration of this exact variant within your mixed-text budget**. The composition papers use substantially longer training. Validate the forecast with three pilot seeds before freezing. [Bhattamishra et al.](https://papers.nips.cc/paper_files/paper/2024/file/3f630b20b7b3ac76d3a0016fe29b6dc0-Paper-Conference.pdf), [Li et al.](https://arxiv.org/html/2503.02854v3)

**3. Fallback**

Use **three states and all five nonidentity permutations**, retaining the same protocol. This weakens difficulty and admits stronger algebraic shortcuts. [Liu et al.](https://arxiv.org/html/2210.10749v2)

**4. Validity risks**

Allow training repetition: depth-1 has only `5×44×14=3,080` prompts, so the existing uniqueness requirement would exhaust it. Exclude evaluation prompts; hold out operator compositions at depths ≥2. Never supervise prefixes deeper than six. [Current deduplication](/pack/frozen/prepare/progen.py:169)

Dense supervision changes the question to **supervised state-composition extrapolation**. NOP counts reveal depth; cancellations and parallel associative composition permit shortcuts, so successful OOD accuracy does not prove serial computation. Report per-depth accuracy, operator-reordering and initial-state interventions, and heuristic floors. If baseline OOD saturates, the task cannot discriminate architectures. [Liu et al.](https://arxiv.org/html/2210.10749v2), [Li et al.](https://arxiv.org/html/2503.02854v3)