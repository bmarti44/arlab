"""Frozen LoRA trainer behind adapt()'s `train`. It runs ONLY in the trusted RUN supervisor (harness.py); the surface
(a sandboxed child process) calls it through the RPC stub in surface_api.py. The surface supplies the data, the loss
mix and the hyper-parameters; this loop owns the optimizer steps, the token budget, the deadline and the adapters.

    adapter = train(examples, config=None, init=None)        # an AdapterRef in the surface, an int id here

examples: a list of dicts, each one of
    {"text": str}                                   next-token loss on every token (cut into max_len windows)
    {"prompt": str, "completion": str}              the evaluation's chat format: system prompt with this world's tool
                                                    names (no transcript), user turn `prompt`, assistant turn
                                                    `completion` + <|im_end|>; loss on the assistant tokens only
  optional keys: "weight" (float, default 1.0), "teacher" (a gen.teacher() ref for exactly this completion, from this
  world: adds teacher_weight * KL(teacher top-k || student) on every assistant token).
config (defaults below): rank (<= 64), alpha, targets (subset of q/k/v/o/gate/up/down_proj), layers (None = all 28),
  lr, epochs, batch_size, max_len (batch_size * max_len <= 32768), warmup, min_lr_frac, weight_decay, grad_clip,
  ce_weight, teacher_weight, kl_base (weight of KL(base || adapted) on generic OASST2 replay rows supplied by the
  harness), replay_rows (rows per step), grad_ckpt.
init: an adapter from an earlier train() call in the same world to continue from (same rank/targets/layers).
Returns the adapter's immutable id (index into this world's private list of (cfg, tensors, stats); the weights never
leave this process until the supervisor saves them); adapt() must return one of these (or None for no adaptation).
Training stops early (and returns what it has) when the world's train-token budget or deadline is reached; the base
weights never change.
Every config field is type- and range-checked (replay_rows: int in [0, 64]). The train-token budget counts processed
positions: batch_size x the padded batch length per step, plus 2 x replay_rows x replay length when kl_base > 0.
A non-finite loss, gradient norm or adapter in ANY train() call marks the whole run invalid (sticky), even if an
earlier finite adapter is returned.
"""
from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import lora
from common import PAD_ID, chat_prefix, chat_user, completion_text, system_text
from engine import BudgetExceeded, Teacher

DEFAULTS = {"rank": 16, "alpha": 32, "targets": list(lora.TARGETS), "layers": None, "lr": 2e-4, "epochs": 1,
            "batch_size": 8, "max_len": 1024, "warmup": 0.05, "min_lr_frac": 0.1, "weight_decay": 0.0, "grad_clip": 1.0,
            "ce_weight": 1.0, "teacher_weight": 1.0, "kl_base": 0.0, "replay_rows": 0, "grad_ckpt": False}
KL_POSITIONS = 64      # replay positions per row used for the KL-to-base term
MAX_REPLAY_ROWS = 64


def _num(cfg, key, lo, hi, lo_open=False):
    x = cfg[key]
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x > hi or x < lo or (lo_open and x == lo):
        raise ValueError(f"config[{key!r}] must be a finite number in {'(' if lo_open else '['}{lo}, {hi}], got {x!r}")
    return float(x)


def _int(cfg, key, lo, hi):
    x = cfg[key]
    if isinstance(x, bool) or not isinstance(x, int) or not lo <= x <= hi:
        raise ValueError(f"config[{key!r}] must be an int in [{lo}, {hi}], got {x!r}")
    return x


def check_config(config) -> dict:
    """Validate every field (types and ranges); returns the full config. Raises ValueError."""
    if config is not None and not isinstance(config, dict):
        raise TypeError("config must be a dict or None")
    cfg = {**DEFAULTS, **(config or {})}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown train config keys {sorted(unknown)}")
    out = {**cfg, **lora.check_config(cfg)}
    out["lr"] = _num(cfg, "lr", 0, 0.1, lo_open=True)
    out["epochs"] = _num(cfg, "epochs", 0, 1000, lo_open=True)
    out["batch_size"] = _int(cfg, "batch_size", 1, 64)
    out["max_len"] = _int(cfg, "max_len", 64, 4096)
    if out["batch_size"] * out["max_len"] > 32_768:
        raise ValueError("batch_size * max_len must be <= 32768 tokens per step (use more steps instead)")
    for k, hi in (("warmup", 1), ("min_lr_frac", 1), ("weight_decay", 1)):
        out[k] = _num(cfg, k, 0, hi)
    out["grad_clip"] = _num(cfg, "grad_clip", 0, 1000, lo_open=True)
    for k in ("ce_weight", "teacher_weight", "kl_base"):
        out[k] = _num(cfg, k, 0, 1000)
    out["replay_rows"] = _int(cfg, "replay_rows", 0, MAX_REPLAY_ROWS)
    if not isinstance(cfg["grad_ckpt"], bool):
        raise ValueError("config['grad_ckpt'] must be a bool")
    return out


class Trainer:
    def __init__(self, model, engine, tool_names: list[str], replay: np.ndarray, budget, seed: int, device: str):
        self.model, self.e, self.budget, self.seed, self.device = model, engine, budget, seed, device
        self.replay = replay
        self.system = chat_prefix(system_text(tool_names))
        self._kept: list[tuple[dict, dict, dict]] = []     # adapter id -> (cfg, tensors, stats) as trained
        self.calls = 0
        self.nan_seen = False        # sticky: any non-finite loss/gradient/adapter in any train() call of this world

    @property
    def produced(self) -> list[int]:
        return list(range(len(self._kept)))

    def _id(self, ad) -> int:
        if isinstance(ad, bool) or not isinstance(ad, int) or not 0 <= ad < len(self._kept):
            raise ValueError("adapt() must return an adapter produced by train() in this world (or None)")
        return ad

    def saved(self, ad: int) -> tuple[dict, dict]:
        """The (cfg, tensors) train() produced under this id."""
        cfg, tens, _ = self._kept[self._id(ad)]
        return cfg, tens

    def discard_last(self):
        """The supervisor drops an adapter whose train() returned too late (harness.LATE_S)."""
        self._kept.pop()

    def saved_stats(self, ad: int) -> dict:
        return dict(self._kept[self._id(ad)][2])

    def _items(self, examples, max_len):
        items, skipped = [], 0
        sys_ids = None
        for ex in examples:
            if not isinstance(ex, dict):
                raise TypeError("each example must be a dict")
            w = ex.get("weight", 1.0)
            if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 or w > 1000:
                raise ValueError("example weight must be finite and >= 0")
            if "text" in ex:
                ids = self.e.encode(str(ex["text"]))
                for s in range(0, max(1, len(ids) - 1), max_len):
                    win = ids[s:s + max_len]
                    if len(win) >= 2:
                        items.append((win, 1, w, None))
                continue
            if sys_ids is None:
                sys_ids = self.e.encode(self.system)
            p = sys_ids + self.e.encode(chat_user(str(ex["prompt"])))
            c = self.e.encode(completion_text(str(ex["completion"])))
            t = ex.get("teacher")
            if t is not None:
                if not isinstance(t, Teacher) or t.completion != ex["completion"] or t.ids.shape[0] != len(c):
                    raise ValueError("example['teacher'] must be gen.teacher() output for this exact completion")
            if len(p) + len(c) > max_len:
                skipped += 1
                continue
            items.append((p + c, len(p), w, t))
        return items, skipped

    def train(self, examples: list[dict], config: dict | None = None, init: int | None = None) -> int:
        if self.budget.seconds_left() <= 0:
            raise BudgetExceeded("deadline passed before train()")
        if not isinstance(examples, list):
            raise TypeError("examples must be a list of dicts")
        cfg = check_config(config)
        lcfg = lora.check_config(cfg)
        max_len, bs = cfg["max_len"], cfg["batch_size"]
        items, skipped = self._items(examples, max_len)
        if not items:
            raise ValueError("no usable training examples")
        if init is not None:
            icfg, itens = self.saved(init)
            if icfg != lcfg:
                raise ValueError("init must be an adapter from this world's train() with the same rank/targets/layers")
        self.calls += 1
        seed = self.seed * 101 + self.calls
        rng = random.Random(seed)
        model = self.model
        assert lora.n_wrapped(model) == 0
        mods = lora.attach(model, lcfg, seed)
        if init is not None:
            with torch.no_grad():
                for k, m in mods.items():
                    m.A.copy_(itens[f"{k}.A"])
                    m.B.copy_(itens[f"{k}.B"])
        params = [p for m in mods.values() for p in (m.A, m.B)]
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        total = max(1, int(cfg["epochs"] * math.ceil(len(items) / bs)))
        if cfg["grad_ckpt"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        steps, tokens, losses, stop, nan = 0, 0, [], None, False
        order: list[int] = []
        try:
            while steps < total:
                if not order:
                    order = list(range(len(items)))
                    rng.shuffle(order)
                batch = [items[i] for i in order[:bs]]
                order = order[bs:]
                # processed positions: the padded batch, plus the replay rows twice (base + adapted forward)
                n_tok = len(batch) * max(len(b[0]) for b in batch)
                if cfg["kl_base"] > 0 and cfg["replay_rows"] > 0:
                    n_tok += 2 * cfg["replay_rows"] * (self.replay.shape[1] - 1)
                if self.budget.seconds_left() <= 0:
                    stop = "deadline"
                    break
                if not self.budget.can_train(n_tok):
                    stop = "train_tokens"
                    break
                self.budget.charge_train(n_tok)
                tokens += n_tok
                frac = steps / total
                lr = cfg["lr"] * (frac / cfg["warmup"] if cfg["warmup"] > 0 and frac < cfg["warmup"] else
                                  cfg["min_lr_frac"] + (1 - cfg["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * frac)))
                for g in opt.param_groups:
                    g["lr"] = lr
                loss = self._loss(batch, cfg, mods, rng)
                if not torch.isfinite(loss):
                    nan, stop = True, "nan"
                    break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
                if not torch.isfinite(gnorm):
                    nan, stop = True, "nan"
                    break
                opt.step()
                losses.append(loss.item())
                steps += 1
        finally:
            model.eval()
            if cfg["grad_ckpt"]:
                model.gradient_checkpointing_disable()
            tens = lora.tensors(mods)
            lora.detach(model)
            del opt
        if not all(torch.isfinite(t).all() for t in tens.values()):
            nan = True
        if nan:
            self.nan_seen = True
        stats = {"steps": steps, "planned_steps": total, "tokens": tokens, "examples": len(items), "skipped": skipped,
                 "loss_first": losses[0] if losses else None, "loss_last": float(np.mean(losses[-10:])) if losses else None,
                 "stopped": stop, "nan": nan, "time": time.time()}
        self._kept.append((copy.deepcopy(lcfg), tens, stats))
        return len(self._kept) - 1

    def _hidden(self, ids, mask):
        return self.model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state

    def _loss(self, batch, cfg, mods, rng):
        d = self.device
        T = max(len(b[0]) for b in batch)
        ids = torch.full((len(batch), T), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(batch), T), dtype=torch.long)
        for r, (x, _, _, _) in enumerate(batch):
            ids[r, :len(x)] = torch.tensor(x)
            mask[r, :len(x)] = 1
        ids, mask = ids.to(d), mask.to(d)
        h = self._hidden(ids, mask)
        ce_num, ce_den, kl_sum, kl_n = 0.0, 0.0, 0.0, 0
        for r, (x, start, w, t) in enumerate(batch):
            # hidden at position j predicts token j+1; targets are tokens start..len-1
            logits = self.model.lm_head(h[r, start - 1:len(x) - 1]).float()
            tgt = ids[r, start:len(x)]
            ce = F.cross_entropy(logits, tgt, reduction="sum")
            ce_num = ce_num + w * ce
            ce_den += w * len(tgt)
            if t is not None and cfg["teacher_weight"] > 0:
                lp = torch.log_softmax(logits, -1)
                t_lp = t.logprobs.to(d).float()
                t_p = torch.softmax(t_lp, -1)                                  # renormalised over the top-k
                s_lp = lp.gather(-1, t.ids.to(d).long())
                kl_sum = kl_sum + w * (t_p * (torch.log_softmax(t_lp, -1) - s_lp)).sum()
                kl_n += w * len(tgt)
        loss = cfg["ce_weight"] * ce_num / max(ce_den, 1e-9)
        if kl_n:
            loss = loss + cfg["teacher_weight"] * kl_sum / kl_n
        if cfg["kl_base"] > 0 and cfg["replay_rows"] > 0:
            rows = [self.replay[rng.randrange(len(self.replay))] for _ in range(int(cfg["replay_rows"]))]
            x = torch.as_tensor(np.stack(rows)[:, :-1].astype(np.int64), device=d)
            m = torch.ones_like(x)
            P = min(KL_POSITIONS, x.shape[1])
            pos = torch.tensor(sorted(rng.sample(range(x.shape[1]), P)), device=d)
            with torch.no_grad(), lora.disabled(mods):
                base_lp = torch.log_softmax(self.model.lm_head(self._hidden(x, m)[:, pos]).float(), -1)
            st_lp = torch.log_softmax(self.model.lm_head(self._hidden(x, m)[:, pos]).float(), -1)
            loss = loss + cfg["kl_base"] * (base_lp.exp() * (base_lp - st_lp)).sum(-1).mean()
        return loss
