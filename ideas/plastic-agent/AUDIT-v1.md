# plastic-agent v1 winner audit (gpt-6-astra, 2026-09-30)

Winner: v1 incumbent 0008. Prompt: /tmp/arlab-review-pa-winner/prompt.md. Verbatim output:

**Invalidating exploits found: none.** The code supports the intended mechanism: it converts observations from the supplied world transcript into supervised goal-to-call examples, then learns those mappings in LoRA weights. However, this is **FauxOS-specific semantic relabeling**, not evidence of general learning in arbitrary new environments. The forgetting guard also masks a real measured GSM8K regression.

I inspected the code and ran read-only, in-memory checks. No files were changed. The supplied artifacts contain no winning weights or evaluation logs, so I cannot independently reproduce the reported accuracies.

1. **The world-specific information reaches evaluation through LoRA, as intended. — Does not invalidate.**

   [winner/adapt.py:98](/winner/adapt.py:98) parses successful transcript calls, identifies operations from observations, and builds examples using those calls. [winner/adapt.py:135](/winner/adapt.py:135) uses the documented prompt formatter; [winner/adapt.py:150](/winner/adapt.py:150) returns one ordinary training result.

   The trainer supplies tool names without the transcript and trains on assistant completion tokens ([trainer.py:94](/pack/frozen/run/trainer.py:94), [trainer.py:134](/pack/frozen/run/trainer.py:134), [trainer.py:249](/pack/frozen/run/trainer.py:249)). Evaluation merges that adapter and supplies only tool names and the goal ([evaluate.py:219](/pack/frozen/eval/evaluate.py:219)).

   On all eight reconstructed holdout worlds and their twins, blank observations yielded no examples. Replacing tool names with opaque names preserved the examples with correspondingly renamed completions.

2. **Handwritten domain knowledge substantially limits the claim, but I found no evidence of hidden-template leakage. — Does not invalidate the documented benchmark; broader transfer claims are unsupported.**

   The operation recognizer and goal generator are hardcoded ([winner/adapt.py:34](/winner/adapt.py:34), [winner/adapt.py:62](/winner/adapt.py:62)). They already know concepts such as archive, clone, and swapping places. What varies and gets learned is principally **which tool implements each recognized operation, argument placement, and certain variants**.

   This approach is explicitly proposed in [program.md:60](/pack/program.md:60). Several exact goal phrasings are disclosed at [program.md:42](/pack/program.md:42); the exact swap wording also appears in agent-visible [IDEA.md:48](/pack/IDEA.md:48), whose visibility is specified at [pack.yaml:46](/pack/pack.yaml:46). The 731/739 ID substitution is ordinary augmentation, expressly permitted at [program.md:64](/pack/program.md:64), not evidence of a hidden object lookup.

   Concrete limitations argue against treating this as a general semantic learner:

   - Count requires two string arguments, while the winner accepts one.
   - Weight observations say `total:`, which its recognizer misses.
   - Tag-search observations lack the `found`/`matching` words it requires.

   Compare [winner/adapt.py:53](/winner/adapt.py:53), [winner/adapt.py:86](/winner/adapt.py:86), and [fauxos.py:217](/pack/frozen/prepare/fauxos.py:217). My reconstruction confirmed **zero training examples for count, weigh, and find_tag whenever present**, in both target and twin worlds. New operations or substantially different observation language would require changes to this surface.

3. **Forbidden inputs, RPC misuse, and exploited budget gaps: none found. — Does not invalidate.**

   The winner has no file, environment, network, seed-inspection, or RPC-internals access. It makes no generation or teacher calls and does not inspect training statistics. `gen.task_prompt` merely formats `Goal: …` ([surface_api.py:75](/pack/frozen/run/surface_api.py:75)).

   Training uses conventional positive settings ([winner/adapt.py:9](/winner/adapt.py:9)); padded positions are charged before each step ([trainer.py:187](/pack/frozen/run/trainer.py:187)). Reconstructed passes produced 47–67 examples, giving an upper bound of **77,184 training positions** at three epochs and length 384—well below 1.5 million. This checks the winner’s usage, not every hypothetical harness vulnerability or its actual runtime.

4. **The twin control supports transcript-specific mapping, but its “share” is not a causal percentage. — Does not invalidate success; limits attribution.**

   Both passes use the same surface and training seed ([harness.py:275](/pack/frozen/run/harness.py:275)). I found no twin detection or deliberate degradation. Every twin tool name is assigned a different operation ([fauxos.py:148](/pack/frozen/prepare/fauxos.py:148)), so successful learning from the twin actively teaches wrong target-world mappings.

   Consequently, twin accuracy below no-adaptation accuracy is expected negative transfer. With your numbers, the reported share is approximately **1.13–1.16**, because it computes `(success − twin)/(success − none)` ([scoring.py:27](/pack/frozen/eval/scoring.py:27)). It demonstrates sensitivity to the correct transcript, but cannot cleanly separate beneficial format learning from semantic learning. It also does not establish discovery of unfamiliar operation types.

5. **The combined forgetting guard hides the GSM8K loss. — Does not invalidate under the declared rules; invalidates an unqualified “no forgetting” claim.**

   GSM8K is evaluated while the target adapter is merged ([evaluate.py:230](/pack/frozen/eval/evaluate.py:230)). Thus, accepting your measurements, the 3–6.5-point decline is a measured adverse effect of deploying these adapters. It represents **6–13 fewer correct answers among 200 questions**. Without generated answers and paired outcomes, I cannot distinguish reasoning degradation from answer-format changes or establish statistical significance.

   The surface trains all seven projection types, defaults to all layers, and enables no replay/KL protection ([winner/adapt.py:9](/winner/adapt.py:9), [trainer.py:43](/pack/frozen/run/trainer.py:43)). Specialization with collateral regression is therefore plausible.

   The evaluator pools 200 GSM8K and 100 guard-world outcomes ([evaluate.py:269](/pack/frozen/eval/evaluate.py:269)). If math drops by \(g\) and guard accuracy improves by \(h\), then:

   \[
   \text{battery drop}=(2g-h)/3.
   \]

   For example, a 6.5-point math loss plus an 18-point guard gain produces a **1.67-point combined improvement**. Only the combined drop is guarded ([pack.yaml:31](/pack/pack.yaml:31)); component drops are reported separately. Text NLL likewise has no degradation threshold. Separate component guards would be needed to prevent this masking.

6. **The 0.58 validation ICL score is not a matched holdout comparison. — Does not invalidate 0.80 holdout success.**

   Validation and holdout use different worlds ([prepare.py:46](/pack/frozen/prepare/prepare.py:46)). The supplied figures therefore do not establish that this adapter beats transcript-in-context on the same holdout. That requires the holdout ICL result.