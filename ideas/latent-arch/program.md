# latent-arch — instructions for the research agent
You make ONE change per call. The runner (not you) runs training/evaluation, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator or the data; do not try.
## Goal
maximize accuracy on held-out multi-step programs with NO written reasoning. Programs are 14 single-assignment
statements, arithmetic mod 100 (`a=37;k=12;b=a+4;c=k-3;d=b+c;...;e?` → `41`); the model must put the answer token
as the argmax of the logits right after `?`, in one forward pass. accuracy = mean of acc_id (queried depth k 1..6,
as in training) and acc_depth (k 7..10: deeper than ANY training program; only a model that composes steps can
answer). Current noise: see sigma in history.md. A change is kept only if it beats the incumbent on one seed and then
clearly (by more than 2·SE) on two fresh seeds. The question: does computing in latent depth (loops, recursion,
adaptive depth, extra latent positions, recurrent/linear-attention state) beat the plain GPT at equal parameters
and equal wall-clock training time?
## What you may change
model.py and train.py: any causal architecture in pure PyTorch and its training recipe (optimizer, LR, schedules over
`progress`, accumulation, loss shaping that does not key on token ids). Keep the API: build(config) -> state,
train_step(state, (x, y), step, progress) -> loss, save(state, path), load(path, device) -> model whose
forward(idx) returns causal logits (B, T, 8192) for the same T. The baseline is nanochat-lite's tuned 6×384 GPT.
## What the code does
The frozen harness starts the clock BEFORE importing train.py; import, build, torch.compile, every train_step and
save count against 330 s of wall clock (GPU synchronized at every deadline check). Each batch is 64×1024 tokens:
58 rows of web text + 6 rows of packed programs (~9 % of tokens), rows shuffled, plain next-token loss.
`progress` = elapsed / budget. Anything that costs more per token (K passes, extra positions, slow scans) sees
proportionally fewer tokens: nothing inside the budget is free. The evaluator loads your checkpoint eagerly
(no torch.compile) and scores raw logits itself.
## Guards (a run that fails one is discarded; invalid runs are worse)
- val_bpb (web text) ≤ 1.03× baseline; acc_id ≥ 0.90× baseline.
- params_m (parameters + floating buffers of the loaded model) ≤ 1.05× baseline (~27.7 M): loops must reuse weights.
- infer_flops_tok (FLOPs counted per token on the actual inputs, so adaptive depth is counted where it happens)
  ≤ 2.0× baseline; infer_s (timed eval forward) ≤ 2.5×; RUN wall time ≤ 1.3×; peak memory ≤ 40 GB.
- Invalid: non-finite loss/logits, wrong logit shape, a non-causal model (text or program rows), any op outside
  torch's aten/prims (no custom Triton/CUDA kernels, no torch.library ops, no fla), train_seconds > 360, weights
  that change after save.
## Forbidden (bounded by rules and diff review)
- No special-casing the format: never branch on specific token ids or hand-code the program syntax (digits,
  `=`, `;`, `?`). The vocabulary is secretly permuted, so hard-coded ids point at random tokens anyway.
  Adaptive compute must be learned from content.
- Train only on the batches the harness hands you: no generating synthetic programs, no reading files, the
  environment or the network, no touching the harness, its clock or its globals.
## Ideas worth trying (IDEA.md)
Prelude/core/coda loops with input injection (Huginn); random K in training (reduces depth-extrapolation variance),
more K at eval; truncated backprop through early iterations to save time; adaptive halting or a token router picking
recursion depth (MoR); depth-wise LoRA / per-iteration scales on a shared block; Gated-DeltaNet or other linear
attention layers mixed 3:1 with full attention, in pure PyTorch (chunked scan); learned pause / continuous-thought
positions inside forward; deep supervision of intermediate iterations. Watch the text guard: fewer tokens seen costs
val_bpb (Saunshi+: loops trade memorization for reasoning).
## Rules
- One change, one hypothesis_tag (reuse an existing tag for the same idea).
- Read history.md: don't repeat a failed idea unless you change it materially. After 3 discards in a row, try something structurally different.
- Respect constraints.md (facts about this machine: GB10 GPU, sm_121, CUDA 13, no FlashAttention-3, shared GPU).
- Prefer simple changes; equal results with less code are better.
- Keep notes.md short and useful to your future self.
- Finish with the JSON object required by the output schema and nothing after it.
