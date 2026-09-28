"""Frozen constants and prompt formats shared by PREPARE, the RUN harness and EVALUATE.

The surface sees this file (it is under /frozen). It contains no world data: the FauxOS generator and simulator live in
frozen/prepare/ (PREPARE only) and in each split's private/ dir (EVALUATE only).
"""
from __future__ import annotations

# Pinned, cached, offline. Qwen3-1.7B (dense, 28 layers, d = 2048), post-trained hybrid-think checkpoint used with
# thinking disabled.
MODEL_DIR = "/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
OASST_FILE = ("/hf/hub/datasets--OpenAssistant--oasst2/snapshots/179dd21fc55192153d94adb0e0ce8f69e222bf75/"
              "2023-11-05_oasst2_ready.trees.jsonl.gz")
GSM8K_FILE = "/hf/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet"

IM_END, EOT = 151645, 151643          # <|im_end|>, <|endoftext|>
STOP_IDS = (IM_END, EOT)
PAD_ID = EOT
MAX_PROGRAM_TOKENS = 256              # one program: <= 12 calls
MAX_GSM8K_TOKENS = 320

SYSTEM = ("You operate FauxOS, a tool system with its own conventions. Available tools: {tools}.\n"
          "Answer each goal with a program: one tool call per line, written as name(arg, ...) with positional "
          "arguments only (integers or double-quoted strings), at most 12 lines, and nothing else. The output of the "
          "last line is your answer; answer(x) outputs x.")
LOG_HEADER = "\n\nLog of an earlier session in this system (calls and their outputs):\n"
GOAL = "Goal: {goal}"
RETRY = "The program failed at line {line}: {obs}\nWrite the corrected program."
GSM8K_SYSTEM = "You are a helpful assistant."
GSM8K_USER = "{q}\nSolve the problem briefly. End with a final line of the form 'Answer: <number>'."
THINK_OFF = "<think>\n\n</think>\n\n"


def render_transcript(events: list[dict]) -> str:
    """The exploration transcript as text: '> call' then the observation, one event per two lines."""
    return "".join(f"> {e['call']}\n{e['obs']}\n" for e in events)


def system_text(tool_names: list[str], transcript: list[dict] | None = None) -> str:
    s = SYSTEM.format(tools=", ".join(tool_names))
    if transcript is not None:
        s += LOG_HEADER + render_transcript(transcript).rstrip("\n")
    return s


def chat_prefix(system: str) -> str:
    """Everything before the first user turn (shared by all tasks of a world -> computed once, KV reused)."""
    return f"<|im_start|>system\n{system}<|im_end|>\n"


def chat_user(user: str) -> str:
    """A user turn followed by the (non-thinking) assistant header; the model continues from here."""
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{THINK_OFF}"


def task_suffix(goal: str) -> str:
    return chat_user(GOAL.format(goal=goal))


def retry_suffix(goal: str, first_output: str, line: int, obs: str) -> str:
    return task_suffix(goal) + first_output + "<|im_end|>\n" + chat_user(RETRY.format(line=line, obs=obs))


def completion_text(text: str) -> str:
    """A training target: the assistant text followed by <|im_end|> (what the model must emit to stop)."""
    return text + "<|im_end|>"
