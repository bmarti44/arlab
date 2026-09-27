"""Whitespace-delimited tokens for a small reverse Polish calculator."""

import math
import re

OPERATORS = frozenset({"+", "-", "*", "/"})
_NUMBER = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)


def tokenize(text):
    """Return operators as strings and numeric literals as finite floats.

    Whitespace separates every token. There is no implicit multiplication,
    punctuation splitting, or comment syntax.
    """
    tokens = []
    for raw in text.split():
        if raw in OPERATORS:
            tokens.append(raw)
            continue
        if _NUMBER.fullmatch(raw) is None:
            raise ValueError("invalid token")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("numeric literal must be finite")
        tokens.append(value)
    return tokens
