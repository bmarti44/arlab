import pytest
from unit_registry import UnitRegistry
from unit_expression import evaluate, convert_lines


def test_scaled_conversion_and_sorted_snapshot():
    r = UnitRegistry()
    assert r.register("m", "length", 1) is None
    r.register("cm", "length", 0.01)
    assert r.convert(250, "cm", "m") == pytest.approx(2.5)
    assert isinstance(r.convert(1, "m", "m"), float)
    assert r.convert(-2, "m", "cm") == pytest.approx(-200)
    names = r.units()
    assert names == ["cm", "m"]
    names.clear()
    assert r.units() == ["cm", "m"]


def test_affine_temperature_expressions():
    r = UnitRegistry()
    r.register("K", "temperature", 1)
    r.register("C", "temperature", 1, 273.15)
    r.register("F", "temperature", 5 / 9, 255.3722222222222)
    assert evaluate(" 32 F->C ", r) == pytest.approx(0, abs=1e-10)
    assert evaluate("100 C -> K", r) == pytest.approx(373.15)
    assert evaluate("-40 C -> F", r) == pytest.approx(-40)


def test_registration_errors_are_atomic():
    r = UnitRegistry()
    r.register("m", "length", 1)
    for args in [("m", "length", 10), ("bad-name", "x", 1),
                 ("2x", "x", 1), ("x", "", 1), ("x", "x", 0),
                 ("x", "x", -1), ("x", "x", float("inf")),
                 ("x", "x", float("nan")), ("x", "x", 1, float("inf"))]:
        with pytest.raises(ValueError):
            r.register(*args)
    assert r.units() == ["m"]
    assert r.convert(3, "m", "m") == 3.0


def test_unknown_incompatible_and_nonfinite_conversions():
    r = UnitRegistry()
    r.register("m", "length", 1)
    r.register("s", "time", 1)
    r.register("huge", "length", 1e308)
    with pytest.raises(KeyError):
        evaluate("1 M -> m", r)
    with pytest.raises(KeyError):
        r.convert(float("inf"), "missing", "m")
    with pytest.raises(ValueError):
        evaluate("1 m -> s", r)
    for value in [float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            r.convert(value, "m", "m")
    with pytest.raises(ValueError):
        r.convert(1e308, "huge", "m")


def test_numeric_grammar_and_rejected_syntax():
    r = UnitRegistry()
    r.register("u", "x", 1)
    for expression, expected in [("+.5 u->u", .5), ("1. u -> u", 1),
                                 ("-2.5e+2 u->u", -250), ("2E-1 u -> u", .2)]:
        assert evaluate(expression, r) == pytest.approx(expected)
    for expression in ["1u->u", "1 u u", "nan u->u", "inf u->u", "1e999 u->u",
                       ". u->u", "1 u->u extra", "1 u->u # comment", "1e u->u"]:
        with pytest.raises(ValueError):
            evaluate(expression, r)


def test_batch_skips_comments_and_preserves_order():
    r = UnitRegistry()
    r.register("m", "length", 1)
    r.register("cm", "length", .01)
    assert convert_lines("# heading\n\n100 cm -> m\n  # skip\n.5 m->cm\n", r) == [1.0, 50.0]
    assert convert_lines("", r) == []
    assert r.units() == ["cm", "m"]


def test_batch_errors_identify_first_physical_line():
    r = UnitRegistry()
    r.register("u", "x", 1)
    for invalid in ["bad syntax", "1 unknown -> u", "1e999 u->u"]:
        with pytest.raises(ValueError) as error:
            convert_lines("# comment\n1 u->u\n\n" + invalid + "\nbad", r)
        assert str(error.value) == "line 4: invalid conversion"
