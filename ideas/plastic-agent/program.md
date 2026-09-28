# plastic-agent — instructions for the research agent
You make ONE change per call. The runner (not you) runs training/evaluation, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator, the tasks or the data; do not try.
## Goal
maximize success: the fraction of held-out tool-use tasks that Qwen3-1.7B solves in never-seen "FauxOS" worlds using
YOUR per-world LoRA, with NO exploration transcript in the prompt. Noise: see sigma in history.md. A change is kept only
if it beats the incumbent on one seed and then clearly (by more than 2·SE) on two fresh seeds.
The question: can a small model consolidate what it saw while exploring an unfamiliar environment into weights ("sleep")
and approach the transcript-in-context arm at a fraction of the prompt tokens? history.md shows the frozen references:
none (no adaptation), icl (the transcript in the prompt: the target), placebo (the baseline LoRA of ANOTHER world).
The baseline is naive next-token LoRA on the raw transcript text.
## What you may change
adapt.py only: adapt(transcript, tool_names, gen, train) -> an adapter returned by train(...), or None. You choose the
training data built from the transcript, the loss mix, the LoRA config and the schedule. Keep ARM = "adapter".
API (read frozen_run/engine.py, frozen_run/trainer.py, frozen_run/common.py):
- transcript: list of {"call": 'name(arg, ...)', "obs": output text or "error <code>"} (~500 calls, ~12k tokens).
- gen.generate(prompts, max_new_tokens, temperature, context=True|False) and gen.teacher(prompts, completions, k,
  context=True): the BASE model, answering in the evaluation's chat format (context=True adds the transcript).
  gen.task_prompt(goal) is the evaluation's user message; gen.transcript_text the ICL arm's log text.
- train(examples, config, init=None): examples {"text"} or {"prompt", "completion"} (+ "weight", "teacher");
  config: rank <= 64, alpha, targets, layers, lr, epochs, batch_size, max_len, ce_weight, teacher_weight, kl_base,
  replay_rows, grad_ckpt, warmup, min_lr_frac, weight_decay, grad_clip. Several train() calls per world are allowed.
## What the code does
Each world is generated from a fresh seed: 20-30 tools with pseudo-word names (some English verbs, some misleading),
random positional-argument orders, units and a conversion constant, three error codes, and hidden objects. A frozen
scripted explorer produced the transcript (systematic calls, error probes, 30% random probes). The harness gives
adapt() one world at a time on a fresh base model (per-world budget: 180 s wall clock, 400k gen tokens, 1.5M train
tokens; when the budget runs out the last adapter trained in that world is kept). The evaluator merges each world's
adapter and asks for a program per goal: one call per line, positional int / "string" literals, <= 12 lines, the last
line's output is the answer, answer(x) outputs x. The simulator runs it from the state where the exploration ended;
after an error the model gets one retry that shows the failing line and its output. Success = no error, exact final
state (for changes) and the right value in the last output (for questions). Goals are like "Archive item 412.",
"Unlock item 305, then permanently delete it.", "What is the tally checksum of <place>?",
"Archive every unlocked <KIND> item that weighs at least 3 <unit>." (thresholds may need the unit conversion).
Reported (not the metric): placebo_success (your LoRA of world i+1 scored on world i) and specific_gain = success −
placebo_success; a gain that is only "format" shows up in both. Also gap_closure vs icl, per-template success s_*.
## Guards (runs that fail one are discarded)
- battery_drop <= 0.02: 200 GSM8K + 100 tasks of another FauxOS world with ITS transcript in context, vs the base model.
- prefill_tokens <= 3000: no transcript in the prompt at evaluation (ARM must stay "adapter").
- peak memory <= 60 GB; the longest per-world adapt() <= 210 s (else invalid); non-finite training loss = invalid.
## Ideas worth trying (ranked in IDEA.md / TTT-DEEP-DIVE.md §4)
1. Hindsight relabeling (the ARC-TTT recipe): turn transcript calls into (goal -> program) pairs in the evaluation's
   format, phrasing the goal from what the observations SHOW (e.g. obs "archived #412" -> "Archive item 412." ->
   that call; an inspect after a silent "ok" reveals the effect). Multi-call goals from consecutive successful calls.
2. Add forward-dynamics pairs: (call -> observation) and "what does <tool> do / what are its arguments" facts.
3. Augmentation: consistent renaming of ids / values, paraphrased goals, several phrasings per demo.
4. Teacher-KL: gen.teacher(..., context=True) on your demos (1.7B is a weak teacher: a supplementary loss).
5. Protect the battery: kl_base + replay_rows, lower rank, fewer layers; also mix in the raw-transcript LM loss.
Also: self-study (gen proposes goals with the transcript in context, keep only programs consistent with the log),
learning rate / epochs / rank / targets, error-recovery demos in the retry format (gen.retry_prompt).
## Rules
- Use only the transcript and what gen / train return. Never read files, the environment or the network from
  adapt.py; never touch the base model or gen/train internals (the harness checks the base weights between worlds);
  relabel only from what the transcript shows: there is no simulator in RUN, and guessing hidden state is not allowed.
- One change, one hypothesis_tag (reuse an existing tag for the same idea).
- Read history.md: don't repeat a failed idea unless you change it materially. After 3 discards in a row, try something structurally different.
- Respect constraints.md (facts about this machine: GB10 GPU, sm_121, unified memory shared with the owner).
- Prefer simple changes; equal results with less code are better.
- Keep notes.md short and useful to your future self.
- Finish with the JSON object required by the output schema and nothing after it.
