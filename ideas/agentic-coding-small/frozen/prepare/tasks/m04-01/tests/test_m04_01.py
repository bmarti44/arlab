import pytest
from polynomial import Polynomial


def test_normalization_and_copy():
    values = [3, 0, 0]
    p = Polynomial(values)
    values[0] = 9
    assert p.coefficients == (3,)
    assert Polynomial(iter([0, 2, 0])).coefficients == (0, 2)
    assert Polynomial([]).coefficients == Polynomial([0, 0]).coefficients == (0,)


def test_addition_cancellation():
    p, q = Polynomial([1, 2, 3]), Polynomial([-1, 4, -3])
    result = p + q
    assert isinstance(result, Polynomial)
    assert result.coefficients == (0, 6)
    assert p.coefficients == (1, 2, 3)
    assert q.coefficients == (-1, 4, -3)
    assert (p + Polynomial([-1])).coefficients == (0, 2, 3)


def test_multiplication_and_zero():
    p, q = Polynomial([1, 1]), Polynomial([-1, 1])
    result = p * q
    assert isinstance(result, Polynomial)
    assert result.coefficients == (-1, 0, 1)
    assert (p * Polynomial([])).coefficients == (0,)
    assert p.coefficients == (1, 1) and q.coefficients == (-1, 1)


def test_evaluation():
    p = Polynomial([3, -2, 0, 1])
    assert [p(x) for x in [-2, 0, 1, 3]] == [-1, 3, 2, 24]
    assert Polynomial([])(100) == 0


def test_string_signs_and_powers():
    assert str(Polynomial([3, -1, 2])) == '2x^2 - x + 3'
    assert str(Polynomial([0, -1, 0, -1])) == '-x^3 - x'
    assert str(Polynomial([-1, 1])) == 'x - 1'
    assert str(Polynomial([1, 0, 1])) == 'x^2 + 1'
    assert str(Polynomial([-1])) == '-1'
    assert str(Polynomial([])) == '0'


def test_unsupported_operands():
    p = Polynomial([1])
    assert p.__add__(2) is NotImplemented
    assert p.__mul__('x') is NotImplemented
    with pytest.raises(TypeError):
        p + 2
    with pytest.raises(TypeError):
        p * 'x'
