"""Evaluate RPN tokens with a fresh operand stack for each call."""

import math
from tokenizer import OPERATORS, tokenize


def _finite_number(value):
    """Normalize an allowed numeric value and reject booleans."""
    if type(value) not in (int, float):
        raise ValueError("expected a number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("number out of range") from error
    if not math.isfinite(result):
        raise ValueError("number must be finite")
    return result


def _apply(operator, left, right):
    """Apply operands in their original stack order."""
    if operator == "+":
        result = left + right
    elif operator == "-":
        result = left - right
    elif operator == "*":
        result = left * right
    else:
        if right == 0:
            raise ZeroDivisionError("division by zero")
        result = left / right
    if not math.isfinite(result):
        raise ValueError("arithmetic result must be finite")
    return result


def evaluate(tokens):
    """Consume an iterable of numeric values and operator strings."""
    stack = []
    for token in tokens:
        if type(token) in (int, float):
            stack.append(_finite_number(token))
            continue
        if not isinstance(token, str) or token not in OPERATORS:
            raise ValueError("unknown operator or invalid token")
        if len(stack) < 2:
            raise ValueError("not enough operands")
        right = stack.pop()
        left = stack.pop()
        stack.append(_apply(token, left, right))
    if len(stack) != 1:
        raise ValueError("expression must leave exactly one result")
    return float(stack[0])


def calculate(text):
    """Tokenize an entire expression before evaluating it."""
    return evaluate(tokenize(text))
