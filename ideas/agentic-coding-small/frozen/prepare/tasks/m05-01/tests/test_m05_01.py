import pytest
from fraction import Fraction


def test_reduction_and_sign():
    assert Fraction(18, -24).as_tuple() == (-3, 4)
    assert Fraction(-18, -24).as_tuple() == (3, 4)
    assert str(Fraction(8, 4)) == "2/1"


def test_zero_and_default():
    assert Fraction(0, -91).as_tuple() == (0, 1)
    assert str(Fraction(-7)) == "-7/1"


def test_invalid_components():
    with pytest.raises(ValueError):
        Fraction(1, 0)
    for pair in [(True, 2), (2, False), (1.0, 2), (1, "2")]:
        with pytest.raises(TypeError):
            Fraction(*pair)


def test_add_without_mutation():
    a, b = Fraction(1, 6), Fraction(1, 3)
    result = a.add(b)
    assert result.as_tuple() == (1, 2)
    assert result is not a and result is not b
    assert a.as_tuple() == (1, 6) and b.as_tuple() == (1, 3)


def test_subtract_and_cancellation():
    a, b = Fraction(2, 3), Fraction(5, 6)
    assert str(a.subtract(b)) == "-1/6"
    assert a.subtract(a).as_tuple() == (0, 1)
    assert a.as_tuple() == (2, 3) and b.as_tuple() == (5, 6)


def test_large_product_and_zero():
    a, b = Fraction(10**40, 3), Fraction(9, 10**39)
    assert a.multiply(b).as_tuple() == (30, 1)
    assert a.multiply(Fraction(0, 7)).as_tuple() == (0, 1)
    assert a.as_tuple() == (10**40, 3) and b.as_tuple() == (9, 10**39)
