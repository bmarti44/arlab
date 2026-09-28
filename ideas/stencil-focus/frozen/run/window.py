"""Frozen prompt window for stencil-focus (agent-visible). The RUN harness and the evaluator use exactly this code.

Per item the harness calls   plan(sentences, current, query, window) -> Plan   on the surface, where
  sentences  list of dicts, one per sentence of sessions 0..s-1 (s = the current session), chronological:
             {id, session, line, speaker ('mentor' | 'mentee' | 'other'), text, tokens, evicted_at_base_window}
             text = one verbatim span of the frozen splitter stencil.focus3.sentences; ids are 0..len-1.
             evicted_at_base_window: the sentence lies entirely before the thread cut of the prompt WITHOUT a
             reminder (stencil's Exp 4C `evicted_mentor_sentences` criterion, applied to every speaker).
  current    the text of the current session s (always inside the window)
  query      what the model is asked to write (e.g. "function that implements merge sort")
  window     a Window (below): token counting, rendering, packing, "what is displaced at reminder budget B"
A Plan is: ids (ordered sentence ids, each at most once), header (index into HEADERS; 1 = no header),
placement ('after_thread' = stencil 4C: after the truncated thread, before the request | 'before_thread' = before
the thread's opening line) and budget (0..1024, the reminder's token cap; the rendered reminder must fit it).
The reminder is HEADERS[header] + one "- <sentence>" line per id. The thread then keeps its newest tokens so the
whole prompt fits W. The prompt is the native Qwen3 single user message with an empty <think> block.
"""
from __future__ import annotations

import functools
import os
import sys
from dataclasses import dataclass, field

MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"      # Qwen/Qwen3-4B, the 4C trunk
TOKENIZER_FILE = f"/hf/hub/models--Qwen--Qwen3-4B/snapshots/{MODEL_REVISION}/tokenizer.json"
W = 3584                          # prompt window (tokens), stencil Exp 4
MAX_BUDGET = 1024                 # largest reminder a plan may use
HEADERS = ("Earlier instructions still in force:",       # 0: stencil 4C LONG_HEADER (baseline)
           "",                                          # 1: no header
           "Coding requirements from earlier sessions:",
           "Reminder of what your mentor asked earlier:",
           "Earlier mentor notes:")
PLACEMENTS = ("after_thread", "before_thread")
STENCIL_SRC = ("/data/public/stencil/src", "/data/validation/public/stencil/src", "/data/pilot/public/stencil/src")


def _stencil():
    try:
        from stencil import focus3, memorycode
    except ImportError:
        src = next((p for p in STENCIL_SRC if os.path.isdir(p)), None)
        if src is None:
            raise
        sys.path.insert(0, src)
        from stencil import focus3, memorycode
    return memorycode, focus3


@functools.lru_cache(maxsize=1)
def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOKENIZER_FILE)


class PlanError(ValueError):
    pass


@dataclass
class Plan:
    ids: list = field(default_factory=list)
    header: int = 0
    placement: str = "after_thread"
    budget: int = 256


def as_dialogue(item: dict) -> dict:
    """A public item in the shape stencil's memorycode helpers expect."""
    return {"context": {"mentor": item["mentor"], "mentee": item["mentee"]},
            "sessions": [{"text": t} for t in item["sessions"]]}


def render(texts: list, header: int = 0) -> str:
    """Header 0 reproduces stencil.memorycode.render_long_reminder byte for byte."""
    if not texts:
        return ""
    h = HEADERS[header]
    return (h + "\n" if h else "") + "\n".join(f"- {t}" for t in texts)


def build_prompt(dialogue: dict, s: int, query: str, reminder: str, placement: str, window: int) -> dict:
    """after_thread: stencil.memorycode.build_long_prompt unchanged. before_thread: the same algorithm with the
    reminder block placed before the thread's opening line."""
    mc, _ = _stencil()
    tok = tokenizer()
    if placement == "after_thread" or not reminder:
        return mc.build_long_prompt(dialogue, s, query, tok, reminder, window)
    mentor = dialogue["context"]["mentor"]
    head = reminder + "\n\n" + f"This is a thread of dialogues between you and your mentor {mentor}:\n"
    tail = (f" \nBased on information provided, write a {query}. Do not provide example usage. You must follow "
            "all the latest coding guidelines provided by your mentor, including any possible updates.")
    frame_tokens = len(tok.encode(mc.chat_prompt(head + tail)).ids)
    full = mc.thread_text(dialogue, list(range(s + 1)))
    thread_ids = tok.encode(full).ids
    budget = max(window - frame_tokens, 0)
    for _ in range(8):
        keep = thread_ids[len(thread_ids) - budget:] if budget else []
        thread = tok.decode(keep, skip_special_tokens=False)
        prompt = mc.chat_prompt(head + thread + tail)
        actual = len(tok.encode(prompt).ids)
        if actual <= window:
            break
        budget -= actual - window
    if actual > window:
        raise ValueError(f"prompt does not fit W={window} after retrims: {actual}")
    return {"prompt": prompt, "prompt_tokens": actual, "thread_tokens_kept": len(keep),
            "thread_tokens_total": len(thread_ids), "frame_tokens": frame_tokens, "cut_chars": max(len(full) - len(thread), 0)}


class Window:
    """Everything about one item's prompt window. Built by the harness; the surface only calls its methods."""

    def __init__(self, item: dict, window: int = W):
        mc, f3 = _stencil()
        self.W, self.item = window, item
        self.dialogue, self.s, self.query = as_dialogue(item), len(item["sessions"]) - 1, item["query"]
        self._full = mc.thread_text(self.dialogue, list(range(self.s + 1)))
        base = build_prompt(self.dialogue, self.s, self.query, "", "after_thread", window)
        self.base_cut, self._frame0 = base["cut_chars"], base["frame_tokens"]
        self._thread_ids = None
        self.sentences, self._ends = [], []
        tok = tokenizer()
        for i in range(self.s):                     # the position walk of stencil's evicted_mentor_sentences
            body = self._full.index(f"\n\n Session {i} \n\n") + len(f"\n\n Session {i} \n\n")
            text, offset = self.dialogue["sessions"][i]["text"], 0
            for li, (speaker, line) in enumerate(mc.speaker_lines(self.dialogue, i)):
                pos = text.find(line, offset)
                if pos < 0:
                    continue
                offset = pos + len(line)
                for start, sent in f3.sentences(line):
                    end = body + pos + start + len(sent)
                    self.sentences.append({"id": len(self.sentences), "session": i, "line": li, "speaker": speaker,
                                           "text": sent, "tokens": len(tok.encode(sent).ids),
                                           "evicted_at_base_window": end <= self.base_cut})
                    self._ends.append(end)

    # ---- helpers for the surface
    def tokens(self, text: str) -> int:
        return len(tokenizer().encode(text).ids)

    def render(self, ids: list, header: int = 0) -> str:
        return render([self.sentences[i]["text"] for i in ids], header)

    def reminder_tokens(self, ids: list, header: int = 0) -> int:
        return self.tokens(self.render(ids, header)) if ids else 0

    def pack_newest_first(self, ids: list, budget: int, header: int = 0) -> list:
        """stencil.memorycode.pack_long over sentence ids: walk `ids` from the end, stop at the first that does not
        fit `budget` tokens (rendered with `header`), return the kept ids in their original order."""
        kept = []
        for i in reversed(ids):
            trial = [i] + kept
            if self.reminder_tokens(trial, header) > budget:
                break
            kept = trial
        return kept

    def displaced(self, reminder_tokens: int) -> list:
        """Ids of sentences lying entirely before the thread cut when the reminder costs `reminder_tokens`
        (after_thread). 0 gives exactly the evicted_at_base_window set; B > 0 is exact up to the one or two tokens
        of the blank line and re-tokenisation at the cut (use build() for the exact prompt of a finished plan)."""
        if reminder_tokens <= 0:
            return [r["id"] for r in self.sentences if r["evicted_at_base_window"]]
        if self._thread_ids is None:
            self._thread_ids = tokenizer().encode(self._full).ids
        budget = max(self.W - self._frame0 - reminder_tokens - 1, 0)
        keep = self._thread_ids[len(self._thread_ids) - budget:] if budget else []
        cut = max(len(self._full) - len(tokenizer().decode(keep, skip_special_tokens=False)), 0)
        return [r["id"] for r, e in zip(self.sentences, self._ends) if e <= cut]

    # ---- used by the harness and the evaluator
    def check(self, plan) -> Plan:
        """Validate a surface result; returns a clean Plan or raises PlanError."""
        if isinstance(plan, dict):
            try:
                plan = Plan(**plan)
            except TypeError as e:
                raise PlanError(str(e)) from None
        if not isinstance(plan, Plan):
            raise PlanError(f"plan() must return a Plan, got {type(plan).__name__}")
        ids = list(plan.ids)
        if not all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(self.sentences) for i in ids):
            raise PlanError("ids must be sentence ids of sessions 0..s-1")
        if len(set(ids)) != len(ids):
            raise PlanError("duplicate sentence id")
        if not isinstance(plan.header, int) or not 0 <= plan.header < len(HEADERS):
            raise PlanError(f"header must be 0..{len(HEADERS) - 1}")
        if plan.placement not in PLACEMENTS:
            raise PlanError(f"placement must be one of {PLACEMENTS}")
        if not isinstance(plan.budget, int) or not 0 <= plan.budget <= MAX_BUDGET:
            raise PlanError(f"budget must be an int in 0..{MAX_BUDGET}")
        if self.reminder_tokens(ids, plan.header) > plan.budget:
            raise PlanError(f"reminder has {self.reminder_tokens(ids, plan.header)} tokens > budget {plan.budget}")
        return Plan(ids=ids, header=plan.header, placement=plan.placement, budget=plan.budget)

    def build(self, plan: Plan) -> dict:
        """The exact prompt for a (checked) plan: {prompt, prompt_tokens, reminder, reminder_tokens, cut_chars, ...}."""
        reminder = self.render(plan.ids, plan.header)
        out = build_prompt(self.dialogue, self.s, self.query, reminder, plan.placement, self.W)
        return {**out, "reminder": reminder, "reminder_tokens": self.reminder_tokens(plan.ids, plan.header)}
