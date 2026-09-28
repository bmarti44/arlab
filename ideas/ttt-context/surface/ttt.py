"""Editable surface: test-time training (TTT) of Qwen3-1.7B on one long document before answering one question.

BASELINE = the simplest TTT: next-token loss on random 128-token spans of the document, updating only the query
projections (q_proj of all 28 layers), 8 Adam steps, then answering with the document still in context. The keys and
values of the document before each span are the base model's (frozen, from the harness's single prefill), so only
the span is recomputed per step (the qTTT recipe's mechanics; its span count, steps and LR are left to tuning).

The frozen harness calls, for every item, on a freshly imported copy of this module:
    adapt(model, ctx) -> None | transformers Cache
  model  the HF Qwen3ForCausalLM (bf16 on the GPU; eval mode; every parameter frozen). Change parameter VALUES
         only (in place); do not add/replace modules or parameters, do not monkeypatch. Forward hooks are allowed
         and are removed after the item. Weights, buffers and hooks are reset by the harness after every item.
  ctx    doc_ids (1, L) and question_ids (1, Q) token tensors, doc_len, device, seed, generator (a torch CPU
         Generator seeded per item: use it for random choices), time_left() (seconds of the TTT budget left),
         prefix_cache(n) (a new cache with the base model's keys/values of document positions [0, n)).
  return None to answer with the harness's document cache (if DOC_IN_CONTEXT) or with no document (if not);
         or return a Cache to answer after it (e.g. the document re-encoded with the adapted weights).
Time from the module import to the end of adapt() (after a device sync) must stay within the per-item budget;
an item over budget scores 0. Answering (greedy, <= 12 tokens) is frozen.
"""
import torch
import torch.nn.functional as F

DOC_IN_CONTEXT = True        # answer with the document's KV cache in context (False: the question alone)
PREFILL_DOC = True           # have the harness prefill the document before adapt() (needed for prefix_cache)
TARGETS = ("q_proj",)        # which weight matrices adapt (suffix match on module names)
STEPS = 8                    # optimizer steps (stops early if the budget runs out)
SPAN = 128                   # tokens per training span
LR = 1e-4                    # Adam on fp32 master copies of the adapted weights
BETAS = (0.9, 0.999)
MARGIN_S = 0.3               # stop starting new steps when less than this is left of the budget


def span_loss(model, ctx, start: int, length: int) -> torch.Tensor:
    """Mean next-token loss on doc[start:start+length], reading the frozen base-model KV of positions < start."""
    ids = ctx.doc_ids[:, start:start + length]
    out = model(input_ids=ids, past_key_values=ctx.prefix_cache(start), use_cache=True)
    logits = out.logits[0, :-1].float()
    return F.cross_entropy(logits, ids[0, 1:])


def adapt(model, ctx):
    if STEPS <= 0:
        return None
    params = [p for n, p in model.named_parameters() if n.endswith(tuple(f"{t}.weight" for t in TARGETS))]
    masters = [p.detach().float().clone() for p in params]
    opt = torch.optim.Adam(masters, lr=LR, betas=BETAS)
    for p in params:
        p.requires_grad_(True)
    hi = ctx.doc_len - SPAN
    for _ in range(STEPS):
        if ctx.time_left() < MARGIN_S:
            break
        start = int(torch.randint(1, hi, (1,), generator=ctx.generator))
        span_loss(model, ctx, start, SPAN).backward()
        for p, m in zip(params, masters):
            m.grad = p.grad.float()
            p.grad = None
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, m in zip(params, masters):
                p.copy_(m)
    for p in params:
        p.requires_grad_(False)
    return None
