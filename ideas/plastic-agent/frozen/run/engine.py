"""Frozen inference engine (HF transformers, no vLLM) and the budgeted base-model service `Gen`.

Engine: batched greedy/sampled decoding and teacher top-k log-probs over a SHARED PREFIX (system prompt + optional
transcript) whose KV is computed once and copied into a static cache per batch; suffixes are left-padded between the
prefix and the suffix with explicit position ids, so every row sees exactly `prefix + its own suffix`.
The evaluator uses the same Engine, so the surface's self-generated data and the evaluation share one format.

Gen and Budget live ONLY in the trusted RUN supervisor (harness.py): the surface runs in a sandboxed child process and
reaches them through the RPC stubs in surface_api.py (same method names and arguments). Every position processed
counts toward the world's gen-token budget: each prefix once, padded prompt blocks, every decode step of every row of
a batch (finished rows too), padded teacher rows. A batch reserves its worst case before it runs, so the cap is never
exceeded; after the world's deadline (CLOCK_MONOTONIC, set by the supervisor) or budget every call raises
BudgetExceeded.
"""
from __future__ import annotations

import math
import time

import torch
from transformers import StaticCache

from common import (GOAL, MAX_PROGRAM_TOKENS, PAD_ID, RETRY, STOP_IDS, chat_prefix, chat_user, completion_text,
                    render_transcript, system_text)


class BudgetExceeded(Exception):
    pass


def _count(n) -> int:
    if type(n) is not int or n < 0:
        raise ValueError(f"budget charge must be a non-negative int, got {n!r}")
    return n


class Budget:
    """Per-world limits owned by the harness: a wall-clock deadline and token caps for gen and train.
    Tokens = processed positions: prefixes, padded prompt blocks, every decode step of every batch row (finished
    rows included) and padded training batches (plus replay rows, twice when KL-to-base runs a base forward)."""

    def __init__(self, seconds: float, gen_tokens: int, train_tokens: int, start: float | None = None):
        self.deadline = (time.monotonic() if start is None else start) + seconds
        self.gen_cap, self.train_cap = gen_tokens, train_tokens
        self.gen_used = self.train_used = 0

    def seconds_left(self) -> float:
        return self.deadline - time.monotonic()

    def charge_gen(self, n: int):
        """Reserve n positions before the work is done; refuses (BudgetExceeded) past the deadline or the cap."""
        n = _count(n)
        if self.seconds_left() <= 0 or self.gen_used + n > self.gen_cap:
            raise BudgetExceeded(f"gen budget: {self.gen_used}+{n} tokens / {self.gen_cap}, {self.seconds_left():.0f}s left")
        self.gen_used += n

    def refund_gen(self, n: int):
        """Return the unused part of a reservation (decode steps that were not run)."""
        n = _count(n)
        if n > self.gen_used:
            raise ValueError("refund larger than the reservation")
        self.gen_used -= n

    def can_train(self, n: int) -> bool:
        return self.seconds_left() > 0 and self.train_used + _count(n) <= self.train_cap

    def charge_train(self, n: int):
        if not self.can_train(n):
            raise BudgetExceeded(f"train budget: {self.train_used}+{n} tokens / {self.train_cap}")
        self.train_used += n


MAX_SLOTS = 262_144      # batch x (prefix + suffix + new tokens) per static cache: ~29 GB of KV for Qwen3-1.7B in bf16


class Engine:
    def __init__(self, model, tok, device: str):
        self.model, self.tok, self.device = model, tok, device
        self.dtype = next(model.parameters()).dtype

    def encode(self, s: str) -> list[int]:
        return self.tok(s, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def prefix_kv(self, ids: list[int]):
        """Per-layer (K, V) of the prefix, batch 1."""
        from transformers import DynamicCache
        cache = DynamicCache()
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        self.model.model(input_ids=x, past_key_values=cache, use_cache=True)
        return [(k, v) for k, v in cache.to_legacy_cache()], len(ids)

    def _cache(self, pkv, L0: int, B: int, total: int) -> StaticCache:
        cache = StaticCache(config=self.model.config, max_cache_len=total)
        pos = torch.arange(L0, device=self.device)
        for i, (k, v) in enumerate(pkv):
            cache.update(k.expand(B, -1, -1, -1).contiguous(), v.expand(B, -1, -1, -1).contiguous(), i, {"cache_position": pos})
        return cache

    def _block(self, L0: int, rows: list[list[int]], total: int):
        """Left-pad rows into one suffix block after the prefix: ids (B, Ls), mask (B, total), positions (B, Ls)."""
        B, Ls = len(rows), max(len(r) for r in rows)
        ids = torch.full((B, Ls), PAD_ID, dtype=torch.long)
        mask = torch.zeros((B, total), dtype=torch.long)
        pos = torch.zeros((B, Ls), dtype=torch.long)
        mask[:, :L0] = 1
        for r, s in enumerate(rows):
            off = Ls - len(s)
            ids[r, off:] = torch.tensor(s, dtype=torch.long)
            mask[r, L0 + off:L0 + Ls] = 1
            pos[r, off:] = L0 + torch.arange(len(s))
        d = self.device
        return ids.to(d), mask.to(d), pos.to(d), Ls

    def _fwd(self, ids, mask, pos, cache_pos, cache):
        return self.model.model(input_ids=ids, attention_mask=mask, position_ids=pos, cache_position=cache_pos,
                                past_key_values=cache, use_cache=True).last_hidden_state

    @torch.no_grad()
    def generate(self, prefix, suffixes: list[list[int]], max_new: int = MAX_PROGRAM_TOKENS, temperature: float = 0.0,
                 seed: int = 0, batch_size: int = 16, budget: Budget | None = None) -> list[list[int]]:
        """Decode each suffix after the shared prefix ((kv, L0) from prefix_kv). Returns generated ids incl. the stop
        token if one was produced. With a budget, each batch first reserves B x (padded prompt block + max_new)
        positions (refused past the cap or deadline), decoding stops with BudgetExceeded at the deadline, and the
        decode steps that were not run are refunded. The batch shrinks automatically so that batch x sequence slots
        stay <= MAX_SLOTS."""
        pkv, L0 = prefix
        longest = max((len(x) for x in suffixes), default=0)
        batch_size = max(1, min(batch_size, MAX_SLOTS // (L0 + longest + max_new)))
        gen = torch.Generator(device="cpu").manual_seed(seed)
        order = sorted(range(len(suffixes)), key=lambda i: -len(suffixes[i]))
        out: list[list[int]] = [[] for _ in suffixes]
        stop = torch.tensor(STOP_IDS, device=self.device)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            rows = [suffixes[i] for i in idx]
            B, Ls = len(rows), max(len(r) for r in rows)
            if budget:
                budget.charge_gen(B * (Ls + max_new))
            total = L0 + Ls + max_new
            cache = self._cache(pkv, L0, B, total)
            ids, mask, pos, Ls = self._block(L0, rows, total)
            h = self._fwd(ids, mask, pos, torch.arange(L0, L0 + Ls, device=self.device), cache)
            nxt_pos = pos[:, -1] + 1
            done = torch.zeros(B, dtype=torch.bool, device=self.device)
            steps = 0
            for t in range(max_new):
                if budget and budget.seconds_left() <= 0:
                    raise BudgetExceeded("deadline reached while generating")
                logits = self.model.lm_head(h[:, -1]).float()
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, -1).cpu()
                    tok = torch.multinomial(probs, 1, generator=gen).squeeze(-1).to(self.device)
                else:
                    tok = logits.argmax(-1)
                tok = torch.where(done, torch.full_like(tok, PAD_ID), tok)
                for r in (~done).nonzero().flatten().tolist():
                    out[idx[r]].append(int(tok[r]))
                done = done | torch.isin(tok, stop)
                if bool(done.all()) or t == max_new - 1:
                    break
                c = L0 + Ls + t
                mask[:, c] = 1
                h = self._fwd(tok[:, None], mask, nxt_pos[:, None], torch.tensor([c], device=self.device), cache)
                nxt_pos = nxt_pos + 1
                steps += 1
            if budget:
                budget.refund_gen(B * (max_new - steps))
        return out

    @torch.no_grad()
    def topk(self, prefix, pairs: list[tuple[list[int], list[int]]], k: int = 20, batch_size: int = 8,
             budget: Budget | None = None):
        """For each (prompt ids, completion ids): top-k (ids, log-probs) predicting every completion token.
        With a budget, each batch is charged B x its padded length before it runs."""
        pkv, L0 = prefix
        longest = max((len(p) + len(c) for p, c in pairs), default=0)
        batch_size = max(1, min(batch_size, MAX_SLOTS // (L0 + longest)))
        res = [None] * len(pairs)
        for s in range(0, len(pairs), batch_size):
            chunk = list(range(s, min(len(pairs), s + batch_size)))
            rows = [pairs[i][0] + pairs[i][1] for i in chunk]
            total = L0 + max(len(r) for r in rows)
            if budget:
                budget.charge_gen(len(rows) * (total - L0))
            cache = self._cache(pkv, L0, len(rows), total)
            ids, mask, pos, Ls = self._block(L0, rows, total)
            h = self._fwd(ids, mask, pos, torch.arange(L0, L0 + Ls, device=self.device), cache)
            for r, i in enumerate(chunk):
                p, c = pairs[i]
                start = Ls - len(rows[r]) + len(p) - 1          # hidden state that predicts completion token 0
                lp = torch.log_softmax(self.model.lm_head(h[r, start:start + len(c)]).float(), -1)
                v, ix = lp.topk(k, -1)
                res[i] = (ix.to(torch.int32).cpu(), v.cpu())
        return res


class Teacher:
    """Top-k teacher log-probs for one completion (opaque to the surface; consumed by train())."""

    def __init__(self, completion: str, ids: torch.Tensor, logprobs: torch.Tensor):
        self.completion, self.ids, self.logprobs = completion, ids, logprobs


class Gen:
    def __init__(self, engine: Engine, tool_names: list[str], transcript: list[dict], budget: Budget, seed: int):
        self._e, self._budget, self._seed = engine, budget, seed
        self.tool_names, self.transcript = list(tool_names), transcript
        self.transcript_text = render_transcript(transcript)
        self._prefix = {}
        self._calls = 0

    # -- helpers
    def task_prompt(self, goal: str) -> str:
        return GOAL.format(goal=goal)

    def retry_prompt(self, line: int, obs: str) -> str:
        return RETRY.format(line=line, obs=obs)

    def n_tokens(self, text: str) -> int:
        return len(self._e.encode(text))

    @property
    def tokens_left(self) -> int:
        return self._budget.gen_cap - self._budget.gen_used

    def seconds_left(self) -> float:
        return self._budget.seconds_left()

    def _pre(self, context: bool):
        if context not in self._prefix:
            ids = self._e.encode(chat_prefix(system_text(self.tool_names, self.transcript if context else None)))
            self._budget.charge_gen(len(ids))
            self._prefix[context] = self._e.prefix_kv(ids)
        return self._prefix[context]

    # -- the two model calls
    def generate(self, prompts: list[str], max_new_tokens: int = MAX_PROGRAM_TOKENS, temperature: float = 0.0,
                 context: bool = True, batch_size: int = 16) -> list[str]:
        if type(max_new_tokens) is not int or not (1 <= max_new_tokens <= 1024):
            raise ValueError("max_new_tokens must be an int in [1, 1024]")
        temperature = float(temperature)
        if not (math.isfinite(temperature) and 0 <= temperature <= 10):
            raise ValueError("temperature must be finite and in [0, 10]")
        if not all(isinstance(p, str) for p in prompts):
            raise TypeError("prompts must be strings")
        pre = self._pre(bool(context))
        self._calls += 1
        outs = self._e.generate(pre, [self._e.encode(chat_user(p)) for p in prompts], max_new_tokens, temperature,
                                self._seed * 7919 + self._calls, max(1, min(64, int(batch_size))), self._budget)
        return [self._e.tok.decode([t for t in o if t not in (151645, 151643)]) for o in outs]

    def teacher(self, prompts: list[str], completions: list[str], k: int = 20, context: bool = True,
                batch_size: int = 8) -> list[Teacher]:
        if len(prompts) != len(completions) or type(k) is not int or not (1 <= k <= 64):
            raise ValueError("prompts/completions must pair up and k must be an int in [1, 64]")
        if not all(isinstance(x, str) for x in list(prompts) + list(completions)):
            raise TypeError("prompts and completions must be strings")
        pre = self._pre(bool(context))
        pairs = [(self._e.encode(chat_user(p)), self._e.encode(completion_text(c))) for p, c in zip(prompts, completions)]
        got = self._e.topk(pre, pairs, k, max(1, min(32, int(batch_size))), self._budget)
        return [Teacher(c, ix, lp) for c, (ix, lp) in zip(completions, got)]
