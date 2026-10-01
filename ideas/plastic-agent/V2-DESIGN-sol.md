Use **both split redesign and an explicit method rule**. No finite benchmark can guarantee generality; v2 should make wording specialization uncompetitive and auditable. I read the requested files; `ctx/` is absent here, so the reported results are supplied evidence. Nothing was modified.

1. **Splits and disclosure.** Keep validation at **4 worlds × 60 tasks**, with different wording families per world. Randomize observation structure, field labels, semantic verbs, goal phrasing and argument order independently. Use understandable language: arbitrary ciphers would test deciphering rather than consolidation.

   Holdout: **16 worlds × 60 tasks**, using disjoint wording families; every world contains six familiar and four reserved operation types. Report familiar/reserved results separately, distinguishing wording transfer from operation transfer. Include different development-only operations in validation so general synthesis receives feedback. Replace reserved types if their definitions have reached the campaign agent. The existing `novel` split is now a useful regression test, but should not be the sole final holdout.

   Preserve identical wording families between target and twin; generate tool names independently of semantics in both. Disclose the interface, budgets, scoring and distribution-shift contract in `program.md`/`IDEA.md`, but remove exact goals, observations and operation catalogs. Freeze private families before campaigning; release only aggregate validation feedback; finalize once.

2. **Rules and metric.** Forbid benchmark-specific semantic recognizers, operation dictionaries and handwritten goal templates, including equivalent catalogs embedded in generation prompts. Allow call/JSON parsing, generic text processing, and schemas inferred from the current transcript by `gen`. Check with source review plus supervisor logs of generation requests/results and training examples. AST/literal scans help triage; they cannot prove compliance.

   Balance six tasks per operation per world; primary score becomes the mean of world-level operation macro averages. Keep standard/novel scores diagnostic: their mean allows compensation, while their minimum creates a noisy bottleneck. Require positive baseline-relative gain on reserved operations at finalization, alongside the overall MES. Retain twin, budget and inference guards. Add separate `gsm8k_drop` and `guardworld_drop` guards—the evaluator already computes both—with preregistered tolerances and paired uncertainty reporting. Improvements elsewhere must not cancel math regression.

3. **Expected outcomes and method.** My planning estimates, conditional on an informative transcript: none **0.05–0.12**, raw-LoRA **0.10–0.18**, ICL **0.45–0.65**; a general consolidator **0.22–0.35**, with **0.40** a stretch. These are forecasts, not literature-derived FauxOS measurements.

   Start with batched, transcript-conditioned generation of diverse goals for observed calls; retain observed calls as completion labels, then train without context. Add tool summaries/dynamics and modest teacher KL/replay.

   Evidence supports this direction, with substantial scale caveats:

   - [SEAL](https://arxiv.org/html/2506.10943v2): Qwen2.5-7B raw-passage training scored 33.5%, ordinary synthetic implications 39.7%, RL-trained edits 47%; prompt-only rewriting reached 49.4%. Try prompting before expensive meta-RL.
   - [Synthetic continued pretraining](https://arxiv.org/html/2409.07431v2): diverse relational representations beat repeated paraphrases, but used 455M synthetic tokens and an 8B student.
   - [Context distillation](https://arxiv.org/abs/2209.15189) directly trains context-free predictions from context-conditioned outputs. [Prompt distillation](https://arxiv.org/html/2412.14964v2) extends this to knowledge injection, supporting cached soft targets.
   - [TTT](https://arxiv.org/html/2411.07279v2) stresses augmentation and objective design; its major results retain demonstrations at inference, so they do not establish transcript-free consolidation.

4. **Minimal build and power.**

   - [fauxos.py](/pack/frozen/prepare/fauxos.py): separate semantics from rendering; typed scorer values; balanced tasks; independent names.
   - New `frozen/prepare/wording.py`: frozen family banks; copy into private evaluator data.
   - [prepare.py](/pack/frozen/prepare/prepare.py): split assignments, reserved operations, shared twin renderers.
   - [evaluate.py](/pack/frozen/eval/evaluate.py): macro/stratum metrics and paired outcomes.
   - [harness.py](/pack/frozen/run/harness.py): audit RPC data.
   - [tests/test_pack.py](/pack/tests/test_pack.py): rendering-invariant gold, family disjointness, coverage, twins.
   - [pack.yaml](/pack/pack.yaml), `program.md`, `IDEA.md`: updated rules, guards, visibility and power.

   With paired item SD **0.60**, seed SD **0.015**, three seeds and **960 items**: SE = **0.0229**; 2.5·SE = **0.0573**, below MES **0.06**. World clustering can invalidate that calculation: also require a paired world-bootstrap SE ≤ **0.024**; otherwise add worlds. Cap adaptation at **160 seconds/pass**: eight validation passes plus existing overhead should fit approximately **10–30 minutes**; verify in the pilot.