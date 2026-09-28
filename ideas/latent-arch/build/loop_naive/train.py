"""latent-arch surface, part 2: optimizer, schedule and the API the frozen harness calls (baseline = m3b's recipe).

API (frozen trainer / evaluator worker):
  build(config) -> state            config: vocab_size, seq_len, batch, device, seed, train_seconds;
                                    state["model"] is the nn.Module whose parameters + buffers are checkpointed
  train_step(state, (x, y), step, progress) -> loss     progress = elapsed wall clock / training budget (0..1)
  make_model(config) -> nn.Module   config: vocab_size, seq_len, device; same architecture, forward(idx) returns
                                    causal logits (B, T, vocab_size); weights come from the checkpoint
The supervisor owns the clock: import, build, torch.compile, every train_step and the checkpoint write count.
m3b's schedules over step/total_steps are re-expressed over progress (identical shapes).
"""
import random

import torch

from model import GPT, LOOPS_TRAIN, model_config

# ---------------------------------------------------------------------------- hyperparameters (m3b keep)
TOTAL_BATCH_SIZE = 2**16  # tokens per optimizer step (one harness batch of 64 x 1024 tokens)
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0
COMPILE = True            # torch.compile on CUDA only (CPU smoke tests run eagerly)


# ---------------------------------------------------------------------------- optimizer (MuonAdamW, single GPU)
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    denom = (exp_avg_sq / (1 - beta2_t ** step_t)).sqrt() + eps_t
    p.add_(exp_avg / denom, alpha=-(lr_t / (1 - beta1_t ** step_t)))


def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X
    beta2 = beta2_t.to(g.dtype)  # NorMuon variance reduction
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    v_norm_new = ((v_mean * red_dim_size) * step_size.float().square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)
    lr, wd = lr_t.to(g.dtype), wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0  # cautious weight decay
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Muon for 2D matrix params, AdamW for the rest. compiled=True uses torch.compile'd fused steps (CUDA)."""

    def __init__(self, param_groups, compiled=False):
        super().__init__(param_groups, defaults={})
        c = (lambda f: torch.compile(f, dynamic=False, fullgraph=True)) if compiled else (lambda f: f)
        self._adamw_fn, self._muon_fn = c(adamw_step_fused), c(muon_step_fused)
        t = lambda: torch.tensor(0.0, dtype=torch.float32, device="cpu")  # 0-D CPU tensors: no recompiles
        self._a = {k: t() for k in ("step", "lr", "beta1", "beta2", "eps", "wd")}
        self._m = {k: t() for k in ("momentum", "lr", "wd", "beta2")}

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            st = self.state[p]
            if not st:
                st.update(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
            st["step"] += 1
            for k, v in (("step", st["step"]), ("lr", group["lr"]), ("beta1", group["betas"][0]), ("beta2", group["betas"][1]),
                         ("eps", group["eps"]), ("wd", group["weight_decay"])):
                self._a[k].fill_(v)
            a = self._a
            self._adamw_fn(p, p.grad, st["exp_avg"], st["exp_avg_sq"], a["step"], a["lr"], a["beta1"], a["beta2"], a["eps"], a["wd"])

    def _step_muon(self, group):
        params = group["params"]
        p = params[0]
        st = self.state[p]
        shape = p.shape
        if "momentum_buffer" not in st:
            st["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=p.dtype, device=p.device)
            sshape = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            st["second_momentum_buffer"] = torch.zeros(sshape, dtype=p.dtype, device=p.device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([q.grad for q in params])
        stacked_params = torch.stack(params)
        self._m["momentum"].fill_(group["momentum"])
        self._m["beta2"].fill_(group["beta2"])
        self._m["lr"].fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._m["wd"].fill_(group["weight_decay"])
        m = self._m
        self._muon_fn(stacked_grads, stacked_params, st["momentum_buffer"], st["second_momentum_buffer"],
                        m["momentum"], m["lr"], m["wd"], m["beta2"], group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            (self._step_adamw if group["kind"] == "adamw" else self._step_muon)(group)


# ---------------------------------------------------------------------------- surface API (called by the frozen harness)
def setup_optimizer(model, compiled):
    matrix_params = list(model.transformer.h.parameters())
    scale = (model.config.n_embd / 768) ** -0.5  # AdamW LRs ∝ 1/√dmodel (tuned at 768)
    adam = dict(kind="adamw", betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0)
    groups = [dict(adam, params=list(model.lm_head.parameters()), lr=UNEMBEDDING_LR * scale),
              dict(adam, params=list(model.transformer.wte.parameters()), lr=EMBEDDING_LR * scale),
              dict(adam, params=list(model.value_embeds.parameters()), lr=EMBEDDING_LR * scale),
              dict(adam, params=[model.resid_lambdas], lr=SCALAR_LR * 0.01),
              dict(adam, params=[model.x0_lambdas], lr=SCALAR_LR, betas=(0.96, 0.95))]
    for shape in sorted({p.shape for p in matrix_params}):
        groups.append(dict(kind="muon", params=[p for p in matrix_params if p.shape == shape], lr=MATRIX_LR,
                           momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=WEIGHT_DECAY))
    opt = MuonAdamW(groups, compiled=compiled)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    return opt


def lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown + (1 - cooldown) * FINAL_LR_FRAC


def build(config):
    torch.set_float32_matmul_precision("high")
    random.seed(config["seed"])
    cfg = model_config(config["vocab_size"], config["seq_len"])
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=config["device"])
    model.init_weights()
    compiled = COMPILE and config["device"] == "cuda"
    opt = setup_optimizer(model, compiled)
    accum = max(1, TOTAL_BATCH_SIZE // (config["batch"] * config["seq_len"]))
    fwd = torch.compile(model, dynamic=False) if compiled else model
    return {"model": model, "fwd": fwd, "opt": opt, "accum": accum, "cfg": cfg, "opt_steps": 0}


def train_step(state, batch, step, progress):
    x, y = batch
    state["model"].k = random.choice(LOOPS_TRAIN)  # loop_naive: random K per step (one compiled graph per K)
    with torch.autocast(x.device.type, dtype=torch.bfloat16):
        loss = state["fwd"](x, y)
    (loss / state["accum"]).backward()
    if (step + 1) % state["accum"] == 0:
        lrm = lr_multiplier(min(progress, 1.0))
        for g in state["opt"].param_groups:
            g["lr"] = g["initial_lr"] * lrm
            if g["kind"] == "muon":
                g["momentum"] = (1 - min(state["opt_steps"] / 300, 1)) * 0.85 + min(state["opt_steps"] / 300, 1) * 0.95
                g["weight_decay"] = WEIGHT_DECAY * (1 - min(progress, 1.0))
        state["opt"].step()
        state["model"].zero_grad(set_to_none=True)
        state["opt_steps"] += 1
    return loss.detach()


def make_model(config):
    """The evaluator's constructor: the same architecture build() trains (weights are then overwritten from the
    data-only checkpoint by frozen code). config: vocab_size, seq_len, device."""
    model = GPT(model_config(config["vocab_size"], config["seq_len"])).to(config["device"])
    model.to_bf16_embeddings()
    return model
