"""nanochat-lite surface: GPT model + MuonAdamW optimizer + training step.

Ported from karpathy/autoresearch train.py (MIT License, Copyright (c) 2026 Andrej Karpathy) with GB10 settings
from mazar/autoresearch-spark (numbers only): SDPA instead of FlashAttention-3, full attention windows,
depth 6 / width 384, 2^17 tokens per optimizer step. The frozen harness owns data, batch loop and token budget;
this file supplies build(config), train_step(state, batch, step, total_steps), save(state, path), load(path, device).
"""
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------- hyperparameters
ASPECT_RATIO = 64        # model_dim = depth * ASPECT_RATIO (rounded up to HEAD_DIM)
HEAD_DIM = 128
DEPTH = 6
TOTAL_BATCH_SIZE = 2**17  # tokens per optimizer step (harness batches are 64 x 1024 = 2^16 tokens)
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0
COMPILE = True


# ---------------------------------------------------------------------------- model
@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 3
    n_kv_head: int = 3
    n_embd: int = 384


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head, self.n_kv_head, self.n_embd = config.n_head, config.n_kv_head, config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:  # value residual (ResFormer) with an input-dependent gate per head
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = norm(apply_rotary_emb(q, cos, sin)), norm(apply_rotary_emb(k, cos, sin))
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
                                           enable_gqa=self.n_kv_head != self.n_head)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin):
        x = x + self.attn(norm(x), ve, cos_sin)
        return x + self.mlp(norm(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({"wte": nn.Embedding(config.vocab_size, config.n_embd),
                                          "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)])})
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(config.vocab_size, kv_dim)
                                           for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            for w in (block.attn.c_q.weight, block.attn.c_k.weight, block.attn.c_v.weight, block.mlp.c_fc.weight):
                torch.nn.init.uniform_(w, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        self.to_bf16_embeddings()

    def to_bf16_embeddings(self):
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._rotary(self.rotary_seq_len, head_dim)
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32, device=device), inv_freq)
        return freqs.cos().bfloat16()[None, :, None, :], freqs.sin().bfloat16()[None, :, None, :]

    def setup_optimizer(self):
        matrix_params = list(self.transformer.h.parameters())
        scale = (self.config.n_embd / 768) ** -0.5  # AdamW LRs ∝ 1/√dmodel (tuned at 768)
        adam = dict(kind="adamw", betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0)
        groups = [dict(adam, params=list(self.lm_head.parameters()), lr=UNEMBEDDING_LR * scale),
                  dict(adam, params=list(self.transformer.wte.parameters()), lr=EMBEDDING_LR * scale),
                  dict(adam, params=list(self.value_embeds.parameters()), lr=EMBEDDING_LR * scale),
                  dict(adam, params=[self.resid_lambdas], lr=SCALAR_LR * 0.01),
                  dict(adam, params=[self.x0_lambdas], lr=SCALAR_LR, betas=(0.96, 0.95))]
        for shape in sorted({p.shape for p in matrix_params}):
            groups.append(dict(kind="muon", params=[p for p in matrix_params if p.shape == shape], lr=MATRIX_LR,
                               momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=WEIGHT_DECAY))
        opt = MuonAdamW(groups)
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]
        return opt

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin)
        softcap = 15
        logits = self.lm_head(norm(x)).float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction=reduction)
        return logits


# ---------------------------------------------------------------------------- optimizer (MuonAdamW, single GPU)
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    denom = (exp_avg_sq / (1 - beta2_t ** step_t)).sqrt() + eps_t
    p.add_(exp_avg / denom, alpha=-(lr_t / (1 - beta1_t ** step_t)))


@torch.compile(dynamic=False, fullgraph=True)
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
    """Muon for 2D matrix params, AdamW for the rest."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
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
            adamw_step_fused(p, p.grad, st["exp_avg"], st["exp_avg_sq"], a["step"], a["lr"], a["beta1"], a["beta2"], a["eps"], a["wd"])

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
        muon_step_fused(stacked_grads, stacked_params, st["momentum_buffer"], st["second_momentum_buffer"],
                        m["momentum"], m["lr"], m["wd"], m["beta2"], group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            (self._step_adamw if group["kind"] == "adamw" else self._step_muon)(group)


# ---------------------------------------------------------------------------- surface API (called by the frozen harness)
def model_config(vocab_size, seq_len):
    dim = ((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    return GPTConfig(sequence_len=seq_len, vocab_size=vocab_size, n_layer=DEPTH, n_head=dim // HEAD_DIM,
                     n_kv_head=dim // HEAD_DIM, n_embd=dim)


def lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown + (1 - cooldown) * FINAL_LR_FRAC


def build(config):
    torch.set_float32_matmul_precision("high")
    cfg = model_config(config["vocab_size"], config["seq_len"])
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=config["device"])
    model.init_weights()
    opt = model.setup_optimizer()
    accum = max(1, TOTAL_BATCH_SIZE // (config["batch"] * config["seq_len"]))
    fwd = torch.compile(model, dynamic=False) if COMPILE else model
    return {"model": model, "fwd": fwd, "opt": opt, "accum": accum, "cfg": cfg, "opt_steps": 0}


def train_step(state, batch, step, total_steps):
    x, y = batch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = state["fwd"](x, y)
    (loss / state["accum"]).backward()
    if (step + 1) % state["accum"] == 0 or step == total_steps - 1:
        progress = step / total_steps
        lrm = lr_multiplier(progress)
        for g in state["opt"].param_groups:
            g["lr"] = g["initial_lr"] * lrm
            if g["kind"] == "muon":
                g["momentum"] = (1 - min(state["opt_steps"] / 300, 1)) * 0.85 + min(state["opt_steps"] / 300, 1) * 0.95
                g["weight_decay"] = WEIGHT_DECAY * (1 - progress)
        state["opt"].step()
        state["model"].zero_grad(set_to_none=True)
        state["opt_steps"] += 1
    return loss.detach()


def save(state, path):
    torch.save({"config": asdict(state["cfg"]), "model": state["model"].state_dict()}, path)


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ck["config"])).to(device)
    model.to_bf16_embeddings()
    model.load_state_dict(ck["model"])
    return model
