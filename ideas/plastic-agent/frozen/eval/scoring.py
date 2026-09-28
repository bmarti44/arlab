"""Frozen scorers used only by EVALUATE (FauxOS task scoring itself lives in the private simulator, fauxos.score)."""
import re


def gsm_answer(s: str):
    """The number on the last 'Answer: <number>' line (commas and a leading $ allowed); None if there is none."""
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", s)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def gsm_score(text: str, gold: int) -> float:
    v = gsm_answer(text)
    return float(v is not None and abs(v - gold) < 1e-6)
