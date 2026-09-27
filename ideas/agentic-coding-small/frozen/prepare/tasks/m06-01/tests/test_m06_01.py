import pytest
from vector2 import Vector2


def test_constructor_defaults_and_floats():
    zero = Vector2()
    assert (zero.x, zero.y) == (0.0, 0.0)
    point = Vector2(-2, 3)
    assert (point.x, point.y) == (-2.0, 3.0)
    assert type(point.x) is float and type(point.y) is float


def test_add_does_not_mutate_inputs():
    a, b = Vector2(1.5, -2), Vector2(-4, 8)
    result = a.add(b)
    assert isinstance(result, Vector2)
    assert (result.x, result.y) == (-2.5, 6.0)
    assert result is not a and result is not b
    assert (a.x, a.y, b.x, b.y) == (1.5, -2.0, -4.0, 8.0)


def test_scale_negative_zero_and_identity():
    v = Vector2(2, -3)
    for factor, coordinates in [(-2, (-4, 6)), (0, (0, 0)), (1, (2, -3))]:
        result = v.scale(factor)
        assert (result.x, result.y) == coordinates
        assert result is not v
    assert (v.x, v.y) == (2, -3)


def test_dot_product():
    a, b = Vector2(2, -3), Vector2(4, 5)
    assert a.dot(b) == -7.0
    assert b.dot(a) == -7.0
    assert isinstance(a.dot(b), float)
    assert Vector2(1, 0).dot(Vector2(0, 2)) == 0.0


def test_point_distance():
    a, b = Vector2(-1, -1), Vector2(2, 3)
    assert a.distance_to(b) == pytest.approx(5.0)
    assert b.distance_to(a) == pytest.approx(5.0)
    assert a.distance_to(a) == 0.0
    assert (a.x, a.y, b.x, b.y) == (-1, -1, 2, 3)


def test_normalization_and_zero_error():
    v = Vector2(3, -4)
    unit = v.normalized()
    assert (unit.x, unit.y) == pytest.approx((0.6, -0.8))
    assert (v.x, v.y) == (3, -4)
    again = unit.normalized()
    assert again is not unit
    assert (again.x, again.y) == pytest.approx((unit.x, unit.y))
    with pytest.raises(ValueError):
        Vector2().normalized()
