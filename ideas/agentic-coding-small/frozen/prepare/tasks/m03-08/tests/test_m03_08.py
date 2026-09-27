from copy import deepcopy
from fractions import Fraction
import pytest
from units import normalize_unit, parse_amount
from recipes import scale_recipe


def test_parse_quantities_exactly():
    cases = [(2, Fraction(2)), (0.1, Fraction(1, 10)), (' 1 3/4 ', Fraction(7, 4)),
             ('03/02', Fraction(3, 2)), ('2.50', Fraction(5, 2)), ('0', Fraction(0)), ('1 5/2', Fraction(7, 2))]
    for value, expected in cases:
        result = parse_amount(value)
        assert type(result) is Fraction and result == expected


def test_reject_bad_quantities():
    assert parse_amount('1/2') == Fraction(1, 2)
    for value in (True, None, [], -1, float('nan'), float('inf'), '-1', '+2', '1e2', '.5', '2.', '1/0', '1  1/2', '1 /2', '', '½'):
        with pytest.raises(ValueError):
            parse_amount(value)


def test_unit_aliases():
    for aliases, expected in [
        ('g gram grams', ('g', 1)), ('kg kilogram kilograms', ('g', 1000)),
        ('ml milliliter milliliters', ('ml', 1)), ('l liter liters', ('ml', 1000)),
        ('tsp teaspoon teaspoons', ('ml', 5)), ('tbsp tablespoon tablespoons', ('ml', 15)),
        ('cup cups', ('ml', 240)), ('each piece pieces', ('each', 1))]:
        for alias in aliases.split():
            assert normalize_unit(' ' + alias.upper() + ' ') == expected
    with pytest.raises(ValueError):
        normalize_unit('oz')


def test_scale_merge_and_keep_first_order():
    ingredients = [
        {'name': ' Flour ', 'amount': '1/2', 'unit': 'kg'},
        {'name': 'Milk', 'amount': 1, 'unit': 'cup'},
        {'name': 'FLOUR', 'amount': 50, 'unit': 'grams'},
        {'name': 'milk', 'amount': 2, 'unit': 'tbsp'},
        {'name': 'egg', 'amount': 1, 'unit': 'piece'}]
    saved = deepcopy(ingredients)
    assert scale_recipe(iter(ingredients), 2, 3) == [
        {'name': 'flour', 'amount': 825.0, 'unit': 'g'},
        {'name': 'milk', 'amount': 405.0, 'unit': 'ml'},
        {'name': 'egg', 'amount': 1.5, 'unit': 'each'}]
    assert ingredients == saved


def test_dimensions_zero_and_internal_spaces():
    rows = [{'name': ' Mix ', 'amount': 0, 'unit': 'g'},
            {'name': 'mix', 'amount': 1, 'unit': 'ml'},
            {'name': 'fine  salt', 'amount': 2, 'unit': 'g'}]
    result = scale_recipe(rows, '1/2', '1 1/2')
    assert result == [{'name': 'mix', 'amount': 0.0, 'unit': 'g'},
                      {'name': 'mix', 'amount': 3.0, 'unit': 'ml'},
                      {'name': 'fine  salt', 'amount': 6.0, 'unit': 'g'}]
    assert all(type(row['amount']) is float for row in result)


def test_round_only_after_aggregation():
    rows = [{'name': 'x', 'amount': '1/3', 'unit': 'g'} for _ in range(3)]
    assert scale_recipe(rows, 1, 1)[0]['amount'] == 1.0
    assert scale_recipe(rows[:1], 1, 1)[0]['amount'] == 0.333333
    ties = [{'name': 'even', 'amount': '0.0000005', 'unit': 'g'},
            {'name': 'odd', 'amount': '0.0000015', 'unit': 'g'}]
    assert [r['amount'] for r in scale_recipe(ties, 1, 1)] == [0.0, 0.000002]


def test_servings_and_ingredient_errors():
    assert scale_recipe([], 1, 2) == []
    for a, b in ((0, 1), (1, 0), (-1, 1), (True, 2), (1, 'bad')):
        with pytest.raises(ValueError):
            scale_recipe([], a, b)
    for row in ({'name': ' ', 'amount': 1, 'unit': 'g'},
                {'name': 'x', 'amount': -1, 'unit': 'g'},
                {'name': 'x', 'amount': 1, 'unit': 'unknown'}):
        with pytest.raises(ValueError):
            scale_recipe([row], 1, 2)
