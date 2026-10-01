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
training data built from the transcript, the loss mix, the LoRA config and the schedule. adapt() receives only the
world being adapted, and its adapter is scored on that world. (The arm is chosen by frozen code, not by adapt.py.)
adapt.py runs in a FRESH sandboxed CPU process per world (frozen_run/child.py): no file but /work and the Python install
is readable, nothing is writable but tempfile's per-world scratch dir (deleted afterwards), no network, no GPU, no
threads or subprocesses (one left running = invalid run). gen / train are RPC stubs (frozen_run/surface_api.py) to the
supervisor, which owns the model, validates and meters every call and keeps every adapter (you get AdapterRef: .id,
.stats). Module state does not survive between worlds; `random` is seeded per world (seed numpy/torch yourself).
API (read frozen_run/surface_api.py, frozen_run/trainer.py, frozen_run/engine.py, frozen_run/common.py):
- transcript: list of {"call": 'name(arg, ...)', "obs": output text or "error <code>"} (120 calls, ~2.6k tokens).
- gen.generate(prompts, max_new_tokens, temperature, context=True|False) and gen.teacher(prompts, completions, k,
  context=True) -> TeacherRefs: the BASE model, in the evaluation's chat format (context=True adds the transcript).
  gen.task_prompt(goal) is the evaluation's user message; gen.transcript_text the ICL arm's log text.
- train(examples, config, init=None) -> AdapterRef: examples {"text"} or {"prompt", "completion"} (+ "weight",
  "teacher": a TeacherRef);
  config: rank <= 64, alpha, targets, layers, lr, epochs, batch_size, max_len, ce_weight, teacher_weight, kl_base,
  replay_rows, grad_ckpt, warmup, min_lr_frac, weight_decay, grad_clip. Several train() calls per world are allowed.
## What the code does
Each world is generated from a fresh seed: 11 tools (inspect + 10 operations on hidden objects: changes and questions)
with pseudo-word names that say nothing about what a tool does, random positional-argument orders, units, error codes.
Every tool prints what it did. **Every world has its own wording**: the format of observations, field names, verbs,
id and list formats, error messages and the phrasing of goals all differ from world to world. Validation and holdout
worlds use DISJOINT wordings, and holdout worlds contain operation types that never occur in validation. Nothing about
the wording or the operation catalog is disclosed: a method that wins must learn each world from its transcript.
A frozen scripted explorer produced the transcript (every tool several times, error probes, 15% random probes). The
harness gives adapt() one world at a time on a fresh base model (per-world budget: 160 s wall clock from before the
import of adapt.py, 400k gen tokens, 1.5M train tokens; when the budget runs out, or at 160 + 15 s when the process is
killed, the last adapter trained in that world is kept; an adapter whose train() returns more than 10 s after the
deadline is discarded). adapt() gets ONE CPU thread: CPU time > 1.1 x wall time + 5 s (native threads count) = invalid.
The evaluator merges each world's adapter and asks for ONE call per goal (a natural-language request in the world's
own phrasing): the first line of the answer that looks like name(arg, ...) is executed (positional int / "string"
literals; a bare word counts as a string; other lines are ignored). The simulator runs it from the state where the
exploration ended; after an error the model gets one retry that shows the error. Success = no error, exact final
state (for changes) and the right value (for questions). Every operation has the same number of tasks per world.
No goal repeats a logged change, and no question is answered by a logged call whose logged answer is still right.
Control: every world has a TWIN with the SAME tool names and argument vocabulary but other semantics (every name does
something else) and its own transcript. adapt() runs on it too (same sandbox and budget; it cannot tell which is which)
and that adapter is scored on the REAL world: twin_success. world_specific_gain = success − twin_success: gains from
name / verb priors or format show up in both. Also gap_closure vs icl and success by op-type stratum.
## Guards (runs that fail one are discarded)
- gsm8k_drop and guardworld_drop (separately; thresholds in history.md: the baseline's mean drop + a margin from its seed-to-seed spread):
  200 GSM8K problems, and 100 tasks of another FauxOS world with ITS transcript in context, each vs the base model.
- prefill_tokens <= 1000: no transcript in the prompt at evaluation (prompt positions per task, retry call included).
- world_specific_share >= 0.5: (success − twin_success) / (success − none_success) (1.0 if that gain < 0.025):
  at least half of your gain over no adaptation must come from what the RIGHT world's transcript shows.
- peak memory <= 60 GB; the longest adapt() pass <= 190 s (else invalid; every pass pays for importing adapt.py);
  a non-finite loss or gradient in ANY train() call = invalid, even if you return an earlier adapter.
- Budgets count processed positions: padded prompt blocks and every decode step of every batch row (gen), padded
  training batches plus 2 x replay_rows x replay length when kl_base > 0 (train). A gen batch reserves its worst case
  (rows x (longest prompt + max_new_tokens)) before it runs; unused decode steps are refunded. Every train config
  field is type- and range-checked (e.g. replay_rows is an int in [0, 64]).
## Ideas worth trying (literature: SEAL self-edits, context distillation, synthetic continued pretraining)
1. Self-generated goals: let the base model (gen.generate with context=True) read the transcript and propose diverse
   natural-language goals that a logged call would achieve; keep the logged call as the completion; train without the
   transcript in context (SEAL / self-study style). Diversity of phrasing matters more than volume.
2. Context distillation: gen.teacher(..., context=True) gives the base model's in-context distribution on your
   prompts; train the context-free model toward it (teacher_weight), alongside the hard labels.
3. Tool summaries the model writes itself (what each tool appears to do, its argument order, its errors) as extra
   training text; forward-dynamics pairs (call -> observation).
4. Protect the batteries: kl_base + replay_rows, lower rank, fewer layers; mix in the raw-transcript LM loss.
Also: learning rate / epochs / rank / targets, error-recovery demos in the retry format (gen.retry_prompt).
## Rules
- Use only the transcript and what gen / train return. Never try to read files, the environment or the network from
  adapt.py, or to get around the sandbox or the RPC stubs (the supervisor records every attempt it can see); treat
  every transcript alike: never try to detect twin worlds (e.g. by verb statistics) to adapt worse on them;
  relabel only from what the transcript shows: there is no simulator in RUN, and guessing hidden state is not allowed.
- **No benchmark-specific recognizers or templates.** Do not hand-write regexes, keyword lists or parsers that map
  observation or goal TEXT to operation types, and no hand-written goal templates or operation catalogs (not in code
  and not inside prompts you give to gen). Allowed: parsing the call syntax name(arg, ...), generic text processing
  (splitting, extracting numbers and quoted strings), and anything the base model infers through gen from the
  current transcript. The supervisor logs every gen / teacher / train request in full (a request that would overflow
  the 24M-char per-world log is refused); kept changes are reviewed against this rule and a violating keep is reverted.
- One change, one hypothesis_tag (reuse an existing tag for the same idea).
- Read history.md: don't repeat a failed idea unless you change it materially. After 3 discards in a row, try something structurally different.
- Respect constraints.md (facts about this machine: GB10 GPU, sm_121, unified memory shared with the owner).
- Prefer simple changes; equal results with less code are better.
- Keep notes.md short and useful to your future self.
- Finish with the JSON object required by the output schema and nothing after it.
