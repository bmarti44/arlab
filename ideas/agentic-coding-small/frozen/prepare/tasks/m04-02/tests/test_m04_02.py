from datetime import datetime, timedelta, timezone
import pytest
from humanizer import Humanizer


def test_whole_seconds_and_subseconds():
    now = datetime(2024, 1, 1)
    h = Humanizer(now)
    for value, expected in [(0, 0), (0.9, 0), (-0.9, 0), (1.9, 1), (-1.9, -1)]:
        assert h.seconds_from(now + timedelta(seconds=value)) == expected
    assert h.describe(now + timedelta(microseconds=999999)) == 'just now'
    assert h.describe(now - timedelta(microseconds=999999)) == 'just now'


def test_unit_boundaries():
    now = datetime(2024, 1, 1)
    h = Humanizer(now)
    for seconds, phrase in [(1, '1 second'), (59, '59 seconds'), (60, '1 minute'),
                            (3599, '59 minutes'), (3600, '1 hour'),
                            (86399, '23 hours'), (86400, '1 day'),
                            (604799, '6 days'), (604800, '1 week')]:
        assert h.describe(now - timedelta(seconds=seconds)) == phrase + ' ago'
        assert h.describe(now + timedelta(seconds=seconds)) == 'in ' + phrase


def test_large_durations_and_floor():
    now = datetime(2024, 6, 1)
    h = Humanizer(now)
    assert h.describe(now - timedelta(days=3, hours=23)) == '3 days ago'
    assert h.describe(now + timedelta(days=365)) == 'in 52 weeks'
    assert h.describe(now + timedelta(seconds=119.99)) == 'in 1 minute'


def test_offsets_represent_instants():
    now = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    h = Humanizer(now)
    east = timezone(timedelta(hours=2))
    assert h.describe(datetime(2024, 1, 1, 14, tzinfo=east)) == 'just now'
    assert h.seconds_from(datetime(2024, 1, 1, 13, tzinfo=east)) == -3600
    assert h.describe(datetime(2024, 1, 1, 13, tzinfo=east)) == '1 hour ago'


def test_iterable_order_and_empty():
    now = datetime(2024, 1, 1)
    h = Humanizer(now)
    moments = [now, now + timedelta(seconds=2), now]
    assert h.describe_many(iter(moments)) == ['just now', 'in 2 seconds', 'just now']
    assert h.describe_many([]) == []


def test_type_and_awareness_errors():
    h = Humanizer(datetime(2024, 1, 1))
    with pytest.raises(TypeError):
        Humanizer('2024-01-01')
    with pytest.raises(TypeError):
        h.describe(0)
    with pytest.raises(TypeError):
        h.describe_many([None])
    with pytest.raises(ValueError):
        h.seconds_from(datetime(2024, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError):
        Humanizer(datetime(2024, 1, 1, tzinfo=timezone.utc)).describe(datetime(2024, 1, 1))
