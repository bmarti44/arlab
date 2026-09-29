"""What adapt() receives: `gen` and `train` are thin RPC stubs in the sandboxed child (child.py). The model, the
tokenizer, the LoRA trainer, every adapter's weights and all budget accounting live in the trusted supervisor process
(harness.py -> engine.Gen / trainer.Trainer). Nothing here can reach them except by asking; every request is
re-validated and metered there. This module is pure Python (no torch).

gen (the BASE model, answering in the evaluation's chat format):
  gen.tool_names, gen.transcript (list of {"call", "obs"}), gen.transcript_text (rendered as in the ICL arm)
  gen.generate(prompts, max_new_tokens=64, temperature=0.0, context=True, batch_size=16) -> list[str]
      prompts are user messages; each is answered with the system prompt holding this world's tool names (plus the
      transcript when context=True). Sampling uses the supervisor's per-world seed.
  gen.teacher(prompts, completions, k=20, context=True, batch_size=8) -> list[TeacherRef]
      top-k log-probs of the base model for every completion token (completion + <|im_end|>), kept in the supervisor;
      attach the returned ref as example["teacher"] (same prompt/completion) for distillation in train().
  gen.task_prompt(goal), gen.retry_prompt(line, obs): the evaluation's user messages
  gen.n_tokens(text) (not metered), gen.tokens_left (gen positions left after the last call), gen.seconds_left()
train(examples, config=None, init=None) -> AdapterRef       (see trainer.py for examples, config and the accounting)
  init: an AdapterRef from an earlier train() call in this world. AdapterRef.id is the supervisor's immutable id,
  AdapterRef.stats a copy of the training stats. adapt() returns one of these refs or None; the supervisor saves ITS
  copy of that adapter (the weights never enter this process).
When the world's deadline or token budget is reached, calls raise BudgetExceeded; if adapt() lets it propagate, the
last adapter trained in this world is kept. Invalid arguments raise ValueError / TypeError.
"""
from __future__ import annotations

import time

from common import GOAL, RETRY, render_transcript


class BudgetExceeded(Exception):
    pass


class AdapterRef:
    __slots__ = ("id", "stats")

    def __init__(self, id: int, stats: dict):
        self.id, self.stats = id, stats

    def __repr__(self):
        return f"AdapterRef({self.id})"


class TeacherRef:
    __slots__ = ("id", "completion")

    def __init__(self, id: int, completion: str):
        self.id, self.completion = id, completion


class _Client:
    """JSON-lines RPC to the supervisor (send one request, wait for its reply)."""

    def __init__(self, send, recv, gen_left: int, train_left: int):
        self._send, self._recv = send, recv
        self.gen_left, self.train_left = gen_left, train_left

    def call(self, fn: str, **args):
        self._send({"op": "call", "fn": fn, "args": args})
        r = self._recv()
        self.gen_left, self.train_left = r.get("gen_left", self.gen_left), r.get("train_left", self.train_left)
        if r.get("ok"):
            return r["result"]
        if r.get("kind") == "budget":
            raise BudgetExceeded(r.get("msg", "budget exceeded"))
        raise (TypeError if r.get("kind") == "type" else ValueError)(r.get("msg", "rejected by the supervisor"))


class Gen:
    def __init__(self, client: _Client, tool_names: list, transcript: list, deadline: float):
        self._c, self._deadline = client, deadline
        self.tool_names, self.transcript = list(tool_names), transcript
        self.transcript_text = render_transcript(transcript)

    def task_prompt(self, goal: str) -> str:
        return GOAL.format(goal=goal)

    def retry_prompt(self, line: int, obs: str) -> str:
        return RETRY.format(line=line, obs=obs)

    def n_tokens(self, text: str) -> int:
        return self._c.call("n_tokens", text=text)

    @property
    def tokens_left(self) -> int:
        return self._c.gen_left

    def seconds_left(self) -> float:
        return self._deadline - time.monotonic()      # CLOCK_MONOTONIC is system-wide: the supervisor's clock

    def generate(self, prompts, max_new_tokens=64, temperature=0.0, context=True, batch_size=16) -> list:
        return self._c.call("generate", prompts=list(prompts), max_new_tokens=max_new_tokens, temperature=temperature,
                            context=context, batch_size=batch_size)

    def teacher(self, prompts, completions, k=20, context=True, batch_size=8) -> list:
        completions = list(completions)
        ids = self._c.call("teacher", prompts=list(prompts), completions=completions, k=k, context=context,
                           batch_size=batch_size)
        return [TeacherRef(i, c) for i, c in zip(ids, completions)]


class Train:
    def __init__(self, client: _Client):
        self._c = client

    def __call__(self, examples, config=None, init=None) -> AdapterRef:
        exs = []
        for ex in examples:
            if not isinstance(ex, dict):
                raise TypeError("each example must be a dict")
            ex = dict(ex)
            if isinstance(ex.get("teacher"), TeacherRef):
                ex["teacher"] = ex["teacher"].id
            exs.append(ex)
        if init is not None and not isinstance(init, AdapterRef):
            raise TypeError("init must be an AdapterRef returned by train() in this world")
        r = self._c.call("train", examples=exs, config=config, init=None if init is None else init.id)
        return AdapterRef(r["id"], r["stats"])
