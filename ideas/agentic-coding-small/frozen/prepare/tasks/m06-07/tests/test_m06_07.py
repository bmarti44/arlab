import pytest
from budget import BudgetTracker


def test_empty_and_limit_only_report():
    tracker = BudgetTracker()
    assert tracker.report('2025-01') == {'month': '2025-01', 'total': 0, 'categories': []}
    assert tracker.set_limit(' Food ', 500) is None
    assert tracker.report('2025-01')['categories'] == [{'category': 'Food', 'spent': 0, 'limit': 500, 'remaining': 500}]


def test_month_filter_sort_and_unlimited_categories():
    tracker = BudgetTracker()
    assert tracker.add('1', '2024-02-29', 'travel', 250) is None
    tracker.add('2', '2024-02-01', ' food ', 100)
    tracker.add('3', '2024-03-01', 'food', 900)
    tracker.add('4', '2023-02-01', 'old', 999)
    assert tracker.report('2024-02') == {'month': '2024-02', 'total': 350, 'categories': [
        {'category': 'food', 'spent': 100, 'limit': None, 'remaining': None},
        {'category': 'travel', 'spent': 250, 'limit': None, 'remaining': None}]}


def test_refunds_zero_net_and_overspending():
    tracker = BudgetTracker()
    tracker.set_limit('food', 50)
    tracker.add('a', '2025-01-01', 'food', 100)
    tracker.add('b', '2025-01-02', 'food', -20)
    tracker.add('c', '2025-01-03', 'misc', 10)
    tracker.add('d', '2025-01-03', 'misc', -10)
    report = tracker.report('2025-01')
    assert report['total'] == 80
    assert report['categories'] == [
        {'category': 'food', 'spent': 80, 'limit': 50, 'remaining': -30},
        {'category': 'misc', 'spent': 0, 'limit': None, 'remaining': None}]


def test_remove_and_reuse_entry_id():
    tracker = BudgetTracker()
    tracker.add('a', '2025-01-01', 'x', 5)
    assert tracker.remove('missing') is False
    assert tracker.remove('a') is True
    assert tracker.remove('a') is False
    tracker.add('a', '2025-02-01', 'y', -5)
    assert tracker.report('2025-01')['categories'] == []
    assert tracker.report('2025-02')['total'] == -5


def test_invalid_adds_leave_ledger_unchanged():
    tracker = BudgetTracker()
    tracker.add('a', '2025-01-01', 'x', 5)
    invalid = [('a', '2025-01-01', 'x', 1), ('', '2025-01-01', 'x', 1), ('b', '2025-02-29', 'x', 1), ('b', '20250101', 'x', 1), ('b', '2025-1-01', 'x', 1), ('b', '0000-01-01', 'x', 1), ('b', '2025-01-01', '  ', 1), ('b', '2025-01-01', 'x', 0)]
    for args in invalid:
        with pytest.raises(ValueError):
            tracker.add(*args)
        assert tracker.report('2025-01')['total'] == 5
    tracker.add('b', '2025-01-01', 'x', 2)
    assert tracker.report('2025-01')['total'] == 7


def test_limit_replacement_validation_and_case():
    tracker = BudgetTracker()
    tracker.set_limit('X', 4)
    tracker.set_limit('X', 0)
    tracker.set_limit('x', 2)
    for category, cents in [('X', -1), (' ', 5)]:
        with pytest.raises(ValueError):
            tracker.set_limit(category, cents)
    rows = tracker.report('2025-01')['categories']
    assert [(row['category'], row['limit']) for row in rows] == [('X', 0), ('x', 2)]


def test_month_validation_and_report_isolation():
    tracker = BudgetTracker()
    tracker.set_limit('x', 10)
    report = tracker.report('2025-01')
    report['categories'][0]['limit'] = 999
    report['categories'].clear()
    assert tracker.report('2025-01')['categories'][0]['limit'] == 10
    for month in ['2025-1', '2025-00', '2025-13', '0000-01', '2025-01-01', ' 2025-01']:
        with pytest.raises(ValueError):
            tracker.report(month)
