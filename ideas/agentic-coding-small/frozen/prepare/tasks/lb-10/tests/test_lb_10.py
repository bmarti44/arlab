import pytest
from tokenizer import OPERATORS, tokenize
from calculator import evaluate, calculate


def test_token_types_signs_and_whitespace():
    tokens = tokenize("  -2\t+.5\n1. 2e-3 + - * / ")
    assert tokens == [-2.0, .5, 1.0, .002, "+", "-", "*", "/"]
    assert all(type(value) is float for value in tokens[:4])
    assert OPERATORS == frozenset({"+", "-", "*", "/"})
    assert tokenize(" \n\t") == []


def test_reject_invalid_lexical_tokens():
    for expression in ["2+3", "nan", "inf", "1e999", ".", "1e", "--2", "2 # x",
                       "(2)", "2 **", "１２"]:
        with pytest.raises(ValueError):
            tokenize(expression)


def test_arithmetic_and_operand_order():
    assert calculate("5 1 2 + 4 * + 3 -") == 14.0
    assert calculate("10 4 - 2 /") == 3.0
    assert calculate("3 2 /") == 1.5
    assert calculate("-2 .5 * 2e1 +") == 19.0
    assert calculate("-7") == -7.0
    assert isinstance(calculate("3"), float)


def test_generators_and_input_preservation():
    tokens = [8, 2, "/", 3, "-"]
    before = tokens[:]
    assert evaluate(tokens) == 1.0
    assert tokens == before
    assert evaluate(iter([2, 3, "+"])) == 5.0
    assert evaluate([4]) == 4.0
    assert isinstance(evaluate([4]), float)


def test_stack_shape_errors_and_fresh_state():
    for expression in ["", "+", "1 +", "1 2", "1 2 + *"]:
        with pytest.raises(ValueError):
            calculate(expression)
    assert calculate("2 3 +") == 5.0


def test_invalid_tokens_and_nonfinite_arithmetic():
    for tokens in [[True], ["2"], [None], [[]], [2, 3, "^"],
                   [float("inf")], [float("nan")], [10 ** 1000], [1e308, 1e308, "+"]]:
        with pytest.raises(ValueError):
            evaluate(tokens)
    with pytest.raises(ValueError):
        calculate("1e308 1e308 *")
    assert evaluate([0]) == 0.0


def test_zero_division_and_lexing_before_evaluation():
    for expression in ["1 0 /", "1 -0.0 /"]:
        with pytest.raises(ZeroDivisionError):
            calculate(expression)
    with pytest.raises(ValueError):
        calculate("1 0 / invalid")
    assert calculate("8 4 /") == 2.0
