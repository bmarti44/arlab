from datetime import date
import pytest
from loans import LoanTracker, parse_date


def test_strict_date_parser():
    assert parse_date('2024-02-29') == date(2024, 2, 29)
    for value in ['2023-02-29', '20240101', '2024-1-01', ' 2024-01-01',
                  '0000-01-01', '2024-13-01', None]:
        with pytest.raises(ValueError):
            parse_date(value)


def test_checkout_dates_and_sorted_snapshot():
    tracker = LoanTracker()
    assert tracker.checkout(' z ', ' Ada ', '2024-02-20') == '2024-03-05'
    assert tracker.checkout('a', 'Bea', '2023-12-31', 1) == '2024-01-01'
    assert tracker.loans() == [('a', 'Bea', '2024-01-01'), ('z', 'Ada', '2024-03-05')]
    tracker.loans().clear()
    assert len(tracker.loans()) == 2


def test_overdue_strict_boundary_and_order():
    tracker = LoanTracker()
    tracker.checkout('z', 'Z', '2024-01-01', 1)
    tracker.checkout('b', 'B', '2024-01-02', 1)
    tracker.checkout('a', 'A', '2024-01-02', 1)
    assert tracker.overdue('2024-01-02') == []
    assert tracker.overdue('2024-01-03') == [('z', 'Z', '2024-01-02')]
    assert tracker.overdue('2024-01-04') == [
        ('z', 'Z', '2024-01-02'), ('a', 'A', '2024-01-03'), ('b', 'B', '2024-01-03')]


def test_renew_from_due_date():
    tracker = LoanTracker()
    tracker.checkout('a', 'A', '2024-02-20', 9)
    assert tracker.renew(' a ', '2024-02-21') == '2024-03-07'
    assert tracker.renew('a', '2024-03-07', 2) == '2024-03-09'
    assert tracker.loans() == [('a', 'A', '2024-03-09')]


def test_return_and_unknown_book():
    tracker = LoanTracker()
    tracker.checkout('a', 'A', '2024-01-01')
    assert tracker.return_book(' a ') == 'A'
    assert tracker.loans() == []
    with pytest.raises(KeyError):
        tracker.return_book('a')
    with pytest.raises(KeyError):
        tracker.renew('a', '2024-01-01')
    assert tracker.checkout('a', 'B', '2024-01-02', 1) == '2024-01-03'


def test_rejected_mutations_are_atomic():
    tracker = LoanTracker()
    tracker.checkout('a', 'A', '2024-01-01', 1)
    with pytest.raises(ValueError):
        tracker.checkout('a', 'B', '2024-01-01')
    with pytest.raises(ValueError):
        tracker.renew('a', '2024-01-03')
    for days in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            tracker.renew('a', '2024-01-02', days)
        with pytest.raises(ValueError):
            tracker.checkout('b', 'B', '2024-01-01', days)
    assert tracker.loans() == [('a', 'A', '2024-01-02')]


def test_identifier_and_date_errors():
    tracker = LoanTracker()
    for book, borrower in [('', 'A'), ('a', '  '), (None, 'A')]:
        with pytest.raises(ValueError):
            tracker.checkout(book, borrower, '2024-01-01')
    with pytest.raises(ValueError):
        tracker.checkout('a', 'A', 'bad')
    with pytest.raises(ValueError):
        tracker.overdue('bad')
    with pytest.raises(ValueError):
        tracker.return_book(' ')
    assert tracker.loans() == []
