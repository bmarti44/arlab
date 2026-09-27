import pytest
from pagination import paginate


def test_first_page_defaults():
    result = paginate(list(range(12)))
    assert result == {
        'items': list(range(10)), 'page': 1, 'per_page': 10, 'total': 12,
        'pages': 2, 'has_previous': False, 'has_next': True,
        'previous_page': None, 'next_page': 2, 'window': [1, 2],
    }


def test_partial_last_page():
    result = paginate('abcde', page=3, per_page=2, radius=2)
    assert result['items'] == ['e']
    assert result['pages'] == 3
    assert result['window'] == [1, 2, 3]
    assert result['has_previous'] is True
    assert result['previous_page'] == 2
    assert result['has_next'] is False
    assert result['next_page'] is None


def test_exact_multiple_and_middle_window():
    result = paginate(range(12), page=2, per_page=3, radius=1)
    assert result['items'] == [3, 4, 5]
    assert result['pages'] == 4
    assert result['window'] == [1, 2, 3]
    assert result['previous_page'] == 1
    assert result['next_page'] == 3


def test_beyond_end_and_empty():
    result = paginate(range(5), page=8, per_page=2, radius=1)
    assert result['items'] == []
    assert result['page'] == 8
    assert result['pages'] == 3
    assert result['previous_page'] == 3
    assert result['window'] == []
    empty = paginate([], page=4, per_page=2, radius=10)
    assert empty['pages'] == 0
    assert empty['items'] == empty['window'] == []
    assert empty['has_previous'] is empty['has_next'] is False
    assert empty['previous_page'] is empty['next_page'] is None


def test_generator_copy_and_zero_radius():
    result = paginate((n for n in range(6)), page=2, per_page=2, radius=0)
    assert result['items'] == [2, 3]
    assert result['window'] == [2]
    original = ['a', 'b', 'c']
    page = paginate(original, per_page=2)
    page['items'].append('x')
    assert original == ['a', 'b', 'c']


def test_invalid_options_and_single_item():
    for option in ['page', 'per_page']:
        for value in [0, -1, True, 1.5, '2']:
            with pytest.raises(ValueError):
                paginate([1], **{option: value})
    for value in [-1, True, 0.5, '0']:
        with pytest.raises(ValueError):
            paginate([1], radius=value)
    assert paginate([7])['items'] == [7]
    assert paginate([7])['pages'] == 1
