"""Frozen answer normalization and exact-match scoring (no LLM judge)."""
import re

STOPS = ("<|im_end|>", "<|endoftext|>")
_NUM = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}


def answer_text(decoded: str) -> str:
    """The first line of the generated text, up to the first stop token (later lines never count)."""
    for s in STOPS:
        decoded = decoded.split(s)[0]
    return decoded.strip().split("\n")[0].strip()


_TOKEN = re.compile(r"(?<![a-z0-9])-?\d+(?:\.\d+)?%?|[a-z0-9]+")


def normalize(s: str) -> str:
    """lowercase; tokens = signed numbers with their % sign, or alphanumeric words (other punctuation separates);
    drop articles; number words -> digits. "-3", "3%" and "3.5" stay different from "3"."""
    toks = _TOKEN.findall(s.lower())
    return " ".join(_NUM.get(t, t) for t in toks if t not in ("a", "an", "the"))


def score(pred: str, aliases: list[str]) -> float:
    """1 iff the normalized answer equals a normalized alias exactly (hedges, sentences, lists score 0)."""
    p = normalize(pred)
    return float(bool(p) and p in {normalize(a) for a in aliases})
