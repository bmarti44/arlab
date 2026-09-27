"""Parse conversion expressions and line-oriented conversion batches."""

import math
import re
from unit_registry import UnitRegistry

_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_UNIT = r"[A-Za-z][A-Za-z0-9_]*"
_EXPRESSION = re.compile(
    rf"\s*({_NUMBER})\s+({_UNIT})\s*->\s*({_UNIT})\s*"
)


def evaluate(expression, registry: UnitRegistry):
    """Evaluate one complete '<number> <source> -> <target>' expression."""
    match = _EXPRESSION.fullmatch(expression)
    if match is None:
        raise ValueError("invalid conversion expression")
    number, source, target = match.groups()
    value = float(number)
    if not math.isfinite(value):
        raise ValueError("number must be finite")
    return registry.convert(value, source, target)


def convert_lines(text, registry: UnitRegistry):
    """Ignore blank/comment lines, attaching physical line numbers to errors."""
    values = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            value = evaluate(line, registry)
        except (ValueError, KeyError) as error:
            raise ValueError(f"line {number}: invalid conversion") from error
        values.append(value)
    return values
