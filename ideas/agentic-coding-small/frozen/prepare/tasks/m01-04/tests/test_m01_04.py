import pytest
from gradebook import Gradebook, weighted_mean

def test_weighted_mean_stream():
    assert weighted_mean((pair for pair in [(10, 1), (40, 3)])) == 32.5
    assert weighted_mean([]) is None
    for weight in (0, -2):
        with pytest.raises(ValueError):
            weighted_mean([(10, weight)])

def test_empty_categories_and_weight_copy():
    weights = {'quiz': 1, 'exam': 3}
    book = Gradebook(weights)
    weights['quiz'] = 99
    assert book.report() == {'quiz': None, 'exam': None}
    assert book.overall() is None
    book.add('quiz', 100)
    assert book.overall() == 100
    book.add('exam', 0)
    assert book.overall() == 25

def test_assignment_percentages():
    book = Gradebook({'work': 2, 'exam': 1})
    book.add('work', 1, 2)
    book.add('work', 90, 100)
    book.add('exam', 1, 1)
    assert book.category_average('work') == 70
    assert book.overall() == pytest.approx(80)

def test_drop_order_and_ties():
    book = Gradebook({'work': 1})
    for score, possible in [(1, 2), (10, 20), (1, 10), (9, 10)]:
        book.add('work', score, possible)
    assert book.drop_lowest('work', 2) == [(1, 10), (1, 2)]
    assert book.category_average('work') == 70
    assert book.drop_lowest('work') == [(10, 20)]

def test_drop_zero_all_and_empty():
    book = Gradebook({'x': 1})
    book.add('x', 30)
    assert book.drop_lowest('x', 0) == []
    assert book.drop_lowest('x', 99) == [(30, 100)]
    assert book.drop_lowest('x') == []
    assert book.overall() is None

def test_validation_preserves_state():
    book = Gradebook({'x': 1})
    book.add('x', 80)
    for score, possible in [(-1, 100), (2, 1), (0, 0), (0, -1)]:
        with pytest.raises(ValueError):
            book.add('x', score, possible)
    with pytest.raises(ValueError):
        book.drop_lowest('x', -1)
    assert book.report() == {'x': 80}
    for weights in ({}, {'x': 0}, {'x': -1}):
        with pytest.raises(ValueError):
            Gradebook(weights)

def test_unknown_and_detached_report():
    book = Gradebook({'b': 1, 'a': 2})
    report = book.report()
    assert list(report) == ['b', 'a']
    report['b'] = 5
    assert book.category_average('b') is None
    for action in (lambda: book.add('z', 2), lambda: book.category_average('z'), lambda: book.drop_lowest('z', -1)):
        with pytest.raises(KeyError):
            action()
