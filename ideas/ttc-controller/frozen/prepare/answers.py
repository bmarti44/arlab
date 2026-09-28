"""Frozen answer extraction and normalization (used by the sampler, PREPARE and the tests; never by RUN).

extract(text, finish) -> the normalized content of the LAST \\boxed{...} of a trace that stopped normally, else None.
normalize(s) -> strip ',', '$', spaces and LaTeX spacing; canonical numbers ('12.0' -> '12', '007' -> '7',
'3.50' -> '3.5', '-0' -> '0'). Anything that is not a plain number is kept verbatim (after stripping), so it can
only match a GSM8K gold (always an integer) if it is one.
"""
import re
from decimal import Decimal, InvalidOperation

_NUM = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)")
_STRIP = ("\\!", "\\,", "\\;", "\\ ", "~", "\\$", "$", ",", " ", "\t", "\n")


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (balanced braces), or None."""
    i = text.rfind("\\boxed")
    while i >= 0:
        j = i + len("\\boxed")
        while j < len(text) and text[j] == " ":
            j += 1
        if j < len(text) and text[j] == "{":
            depth, k = 0, j
            while k < len(text):
                depth += {"{": 1, "}": -1}.get(text[k], 0)
                if depth == 0:
                    return text[j + 1:k]
                k += 1
        i = text.rfind("\\boxed", 0, i)  # unbalanced or malformed: try the previous one
    return None


def normalize(s: str | None) -> str | None:
    if s is None:
        return None
    for ch in _STRIP:
        s = s.replace(ch, "")
    s = s.rstrip(".")
    if _NUM.fullmatch(s):
        try:
            d = Decimal(s)
        except InvalidOperation:
            return s or None
        if d == d.to_integral_value():
            return str(int(d))
        return format(d.normalize(), "f")
    return s or None


def extract(text: str, finish: str) -> str | None:
    if finish != "stop":
        return None
    return normalize(last_boxed(text))
