from datetime import date
import pytest
from booking import BookingCalendar, parse_span


def test_span_validation():
    assert parse_span('2024-02-29', '2024-03-01') == (date(2024, 2, 29), date(2024, 3, 1))
    for start, end in [('2023-02-29', '2023-03-01'), ('20240101', '2024-01-02'),
                       ('2024-1-01', '2024-01-02'), ('2024-01-02', '2024-01-02'),
                       ('2024-01-03', '2024-01-02')]:
        with pytest.raises(ValueError):
            parse_span(start, end)


def test_half_open_and_other_rooms():
    c = BookingCalendar(['B', 'A'])
    assert c.book('one', 'A', '2024-04-01', '2024-04-03') is None
    c.book('two', 'A', '2024-04-03', '2024-04-05')
    c.book('three', 'B', '2024-04-01', '2024-04-03')
    assert c.available('2024-04-02', '2024-04-03') == []
    assert c.available('2024-04-03', '2024-04-04') == ['B']
    assert c.available('2024-04-05', '2024-04-06') == ['A', 'B']


def test_overlap_shapes_are_atomic():
    c = BookingCalendar(['A'])
    c.book('base', 'A', '2024-05-10', '2024-05-20')
    before = c.bookings()
    for start, end in [('09', '11'), ('11', '19'), ('19', '21'), ('09', '21'), ('10', '20')]:
        with pytest.raises(ValueError):
            c.book('attempt', 'A', '2024-05-' + start, '2024-05-' + end)
        assert c.bookings() == before
    c.book('attempt', 'A', '2024-05-20', '2024-05-21')
    assert len(c.bookings()) == 2


def test_ids_rooms_and_cancel():
    c = BookingCalendar(['A', 'B'])
    c.book('id', 'A', '2024-01-01', '2024-01-02')
    with pytest.raises(ValueError):
        c.book('id', 'B', '2024-01-04', '2024-01-05')
    with pytest.raises(ValueError):
        c.book('new', 'C', '2024-01-04', '2024-01-05')
    with pytest.raises(KeyError):
        c.cancel('absent')
    assert len(c.bookings()) == 1
    assert c.cancel('id') is None
    c.book('id', 'B', '2024-01-01', '2024-01-02')
    assert c.available('2024-01-01', '2024-01-02') == ['A']


def test_order_filter_and_detached_lists():
    rooms = ['B', 'A']
    c = BookingCalendar(iter(rooms))
    rooms.append('C')
    c.book('z', 'B', '2024-02-02', '2024-02-03')
    c.book('y', 'A', '2024-02-02', '2024-02-03')
    c.book('x', 'B', '2024-02-01', '2024-02-02')
    assert [r[0] for r in c.bookings()] == ['x', 'y', 'z']
    assert c.bookings('A') == [('y', 'A', '2024-02-02', '2024-02-03')]
    result = c.bookings()
    result.clear()
    assert len(c.bookings()) == 3
    assert c.available('2024-03-01', '2024-03-02') == ['A', 'B']
    with pytest.raises(ValueError):
        c.bookings('C')


def test_invalid_span_does_not_reserve_id():
    c = BookingCalendar(['A'])
    with pytest.raises(ValueError):
        c.book('id', 'A', '2024-04-02', '2024-04-01')
    assert c.bookings() == []
    c.book('id', 'A', '2024-04-01', '2024-04-02')
    with pytest.raises(ValueError):
        c.available('2024-04-01', 'bad')


def test_empty_calendar():
    c = BookingCalendar([])
    assert c.bookings() == []
    assert c.available('2024-01-01', '2024-01-02') == []
    with pytest.raises(ValueError):
        c.available('2024-01-02', '2024-01-01')
    with pytest.raises(ValueError):
        c.book('id', 'A', '2024-01-01', '2024-01-02')
