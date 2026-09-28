"""Frozen scorers used only by EVALUATE (FauxOS task scoring itself lives in the private simulator, fauxos.score)."""
import re


_ANSWER_LINE = re.compile(r"^\s*[*_#]*\s*answer\s*[*_]*\s*:\s*[*_]*\s*(.*?)\s*[*_]*\s*$", re.I)
_NUMBER = re.compile(r"\$?\s*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\.?")


def gsm_answer(s: str):
    """The number on the LAST 'Answer: ...' line; that line's rest must be exactly one number (optional $, thousands
    commas, decimals, a trailing period). Anything else on the line (42e9, 42/7, 'not 42', two numbers) -> None."""
    lines = [m.group(1) for m in map(_ANSWER_LINE.match, s.splitlines()) if m]
    if not lines:
        return None
    m = _NUMBER.fullmatch(lines[-1])
    return float(m.group(1).replace(",", "")) if m else None


def gsm_score(text: str, gold: int) -> float:
    v = gsm_answer(text)
    return float(v is not None and abs(v - gold) < 1e-6)
