"""Frozen LoRA trainer: what adapt() receives as `train`. The surface supplies the data, the loss mix and the
hyper-parameters; this loop owns the optimizer steps, the token budget and the deadline.

    adapter = train(examples, config=None, init=None)

examples: a list of dicts, each one of
    {"text": str}                                   next-token loss on every token (cut into max_len windows)
    {"prompt": str, "completion": str}              the evaluation's chat format: system prompt with this world's tool
                                                    names (no transcript), user turn `prompt`, assistant turn
                                                    `completion` + <|im_end|>; loss on the assistant tokens only
  optional keys: "weight" (float, default 1.0), "teacher" (a gen.teacher() result for exactly this prompt/completion:
  adds teacher_weight * KL(teacher top-k || student) on every assistant token).
config (defaults below): rank (<= 64), alpha, targets (subset of q/k/v/o/gate/up/down_proj), layers (None = all 28),
  lr, epochs, batch_size, max_len (batch_size * max_len <= 32768), warmup, min_lr_frac, weight_decay, grad_clip,
  ce_weight, teacher_weight, kl_base (weight of KL(base || adapted) on generic OASST2 replay rows supplied by the
  harness), replay_rows (rows per step), grad_ckpt.
init: an adapter from an earlier train() call in the same world to continue from (same rank/targets/layers).
Returns an opaque adapter; adapt() must return one of these (or None for no adaptation). Training stops early (and
returns what it has) when the world's train-token budget or deadline is reached; the base weights never change.
"""
from __future__ import annotations

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


class Adapter:
    """Opaque result of train(): LoRA tensors on the CPU plus bookkeeping."""

    def __init__(self, cfg: dict, tensors: dict, stats: dict):
        self.cfg, self.tensors, self.stats = cfg, tensors, stats


class Trainer:
    def __init__(self, model, engine, tool_names: list[str], replay: np.ndarray, budget, seed: int, device: str):
        self.model, self.e, self.budget, self.seed, self.device = model, engine, budget, seed, device
        self.replay = replay
        self.system = chat_prefix(system_text(tool_names))
        self.produced: list[Adapter] = []
        self.calls = 0

    def _items(self, examples, max_len):
        items, skipped = [], 0
        sys_ids = None
        for ex in examples:
            if not isinstance(ex, dict):
                raise TypeError("each example must be a dict")
            w = float(ex.get("weight", 1.0))
            if not math.isfinite(w) or w < 0:
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

    def train(self, examples: list[dict], config: dict | None = None, init: Adapter | None = None) -> Adapter:
        if self.budget.seconds_left() <= 0:
            raise BudgetExceeded("deadline passed before train()")
        cfg = {**DEFAULTS, **(config or {})}
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown train config keys {sorted(unknown)}")
        lcfg = lora.check_config(cfg)
        max_len = int(min(max(64, cfg["max_len"]), 4096))
        bs = int(min(max(1, cfg["batch_size"]), 64))
        if bs * max_len > 32_768:
            raise ValueError("batch_size * max_len must be <= 32768 tokens per step (use more steps instead)")
        items, skipped = self._items(examples, max_len)
        if not items:
            raise ValueError("no usable training examples")
        self.calls += 1
        seed = self.seed * 101 + self.calls
        rng = random.Random(seed)
        model = self.model
        assert lora.n_wrapped(model) == 0
        mods = lora.attach(model, lcfg, seed)
        if init is not None:
            if init not in self.produced or init.cfg != lcfg:
                raise ValueError("init must be an adapter from this world's train() with the same rank/targets/layers")
            with torch.no_grad():
                for k, m in mods.items():
                    m.A.copy_(init.tensors[f"{k}.A"])
                    m.B.copy_(init.tensors[f"{k}.B"])
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
                n_tok = sum(len(b[0]) for b in batch) + 2 * cfg["replay_rows"] * (self.replay.shape[1] if cfg["kl_base"] else 0)
                if self.budget.seconds_left() <= 0:
                    stop = "deadline"
                    break
                if self.budget.train_used + n_tok > self.budget.train_cap:
                    stop = "train_tokens"
                    break
                self.budget.train_used += n_tok
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
                torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
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
        stats = {"steps": steps, "planned_steps": total, "tokens": tokens, "examples": len(items), "skipped": skipped,
                 "loss_first": losses[0] if losses else None, "loss_last": float(np.mean(losses[-10:])) if losses else None,
                 "stopped": stop, "nan": nan, "time": time.time()}
        ad = Adapter(lcfg, tens, stats)
        self.produced.append(ad)
        return ad

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
