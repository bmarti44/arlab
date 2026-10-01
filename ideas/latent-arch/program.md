# latent-arch — instructions for the research agent
You make ONE change per call. The runner (not you) runs training/evaluation, git and the ledger.
You cannot run the experiment yourself and cannot see the evaluator or the data; do not try.
## Goal
maximize accuracy on held-out multi-step state tracking with NO written reasoning. A record is
`= s0 t1 … t14 ?`: a start state (one of 5) and 14 slots, k of which hold an operator (one of 44 fixed-point-free
permutations of the 5 states), the rest a no-op `-`. The answer is s0 pushed through the active operators in order;
the model must put it as the argmax of the logits at `?`, in one forward pass. accuracy = mean of acc_id (k 1..6,
as in training) and acc_depth (k 7..10: deeper than ANY training record; only a model that composes steps can
answer). Current noise: see sigma in history.md. A change is kept only if it beats the incumbent on one seed and then
clearly (by more than 2·SE) on two fresh seeds. The question: does computing in latent depth (loops, recursion,
adaptive depth, extra latent positions, recurrent/linear-attention state) beat the plain GPT at equal parameters
and equal wall-clock training time?
## What you may change
model.py and train.py: any causal architecture in pure PyTorch and its training recipe (optimizer, LR, schedules over
`progress`, accumulation, loss shaping that does not key on token ids). Keep the API: build(config) -> state with
state["model"] = the nn.Module that is checkpointed; train_step(state, (x, y), step, progress) -> loss;
make_model(config) -> the same architecture (config: vocab_size, seq_len, device) whose forward(idx) returns causal
logits (B, T, 8192) for the same T. The baseline is nanochat-lite's tuned 6×384 GPT.
## What the code does
A frozen supervisor starts the clock BEFORE train.py is imported (in a separate trainer process); import, build,
torch.compile, every train_step and the checkpoint write count against 330 s of wall clock. The trainer stops at
the deadline; the supervisor kills it 30 s later no matter what, and then the run is invalid. Each batch is 64×1024
tokens: 48 rows of web text (next-token targets) + 16 rows of packed records (k follows a fixed curriculum: 1, then
1..2, then 1..6), rows shuffled. Record targets are the state after each active operator (at its position) and the
final state at `?`; every other record target is -1: your loss must ignore -1 (the baseline uses ignore_index=-1).
`progress` = elapsed / budget. Anything that costs more per token (K passes, extra positions, slow scans) sees
proportionally fewer tokens: nothing inside the budget is free. The checkpoint is written by frozen code: every
parameter and buffer of state["model"] (nothing else survives). The evaluator rebuilds the model with make_model(),
loads those tensors, and runs it eagerly in a sandboxed process that only receives token ids and cannot read any
file except the Python install and your two files; it scores the returned logits itself.
## Guards (a run that fails one is discarded; invalid runs are worse)
- val_bpb (web text) ≤ 1.03× baseline; acc_id ≥ 0.90× baseline.
- params_m (every float element of every checkpointed tensor, buffers included; non-float tensors count per byte)
  ≤ 1.05× baseline (~27.7 M), and params_bytes (total checkpoint bytes) ≤ 1.05×: loops must reuse weights, and
  packing weights into other dtypes gains nothing. ALL tensor state must be registered parameters or buffers:
  a tensor reachable from the model, its modules or your code that is not one of them makes the run invalid,
  and every registered tensor's storage must be fully covered by registered tensors (no private tail of a buffer).
- infer_flops_tok (FLOPs counted per token on the actual inputs, so adaptive depth is counted where it happens;
  matmul/attention by formula, every other op that reads tensors — elementwise, reductions, gathers — at least its
  element count) ≤ 2.0× baseline; infer_s (timed eval forward) ≤ 2.5×; RUN wall time ≤ 1.3×; peak memory ≤ 40 GB.
- Invalid: non-finite loss/logits, wrong logit shape, a non-causal model (checked at random cut points over the whole 1024-token row), model tensors that change
  during evaluation (eval()/forward must not modify weights or buffers), any op outside
  torch's aten/prims (no custom Triton/CUDA kernels, no torch.library ops, no fla), a trainer killed at the
  deadline, a checkpoint that changed after the deadline.
## Forbidden (bounded by rules and diff review)
- No special-casing the format: never branch on specific token ids or hand-code the record syntax (states,
  operators, `-`, `=`, `?`). The vocabulary is secretly permuted, so hard-coded ids point at random tokens anyway.
  Adaptive compute must be learned from content.
- No symbolic computation: never parse, interpret or execute the programs outside learned network computation
  (no hand-written interpreter, permutation tables, lookup tables keyed on inputs, or decoding hidden states back to
  tokens to reason over them), in training or in forward.
- forward must be a pure function of its input: no caches or state carried across calls, no timing- or
  shape-dependent special paths, no kernels launched outside torch ops (FLOPs must be countable).
  No threads or processes in make_model/forward (invalid: FLOPs are metered on the calling thread). Ops without
  a FLOP formula that are not elementwise/reduction/data-movement are rejected (the error names them).
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
