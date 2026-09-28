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


def normalize(s: str) -> str:
    """lowercase, punctuation -> space, drop articles, number words -> digits, collapse whitespace."""
    toks = re.sub(r"[^a-z0-9]+", " ", s.lower()).split()
    return " ".join(_NUM.get(t, t) for t in toks if t not in ("a", "an", "the"))


def score(pred: str, aliases: list[str]) -> float:
    """1 iff the normalized answer equals a normalized alias exactly (hedges, sentences, lists score 0)."""
    p = normalize(pred)
    return float(bool(p) and p in {normalize(a) for a in aliases})
