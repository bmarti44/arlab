import pytest
from temperature import convert, parse_reading
from templog import TemperatureLog


def test_parse_decimal_and_unit_syntax():
    assert parse_reading(' +12. f ') == (12.0, 'F')
    assert parse_reading('-.5C') == (-0.5, 'C')
    assert parse_reading('0\tK') == (0.0, 'K')
    for text in ['1e2C', 'NaNC', 'inf K', '12', '12°C', '1 C extra', None, '1\nC']:
        with pytest.raises(ValueError):
            parse_reading(text)


def test_conversions_and_absolute_zero():
    assert convert(32, 'f', 'c') == pytest.approx(0)
    assert convert(100, 'C', 'F') == pytest.approx(212)
    assert convert(273.15, 'K', 'C') == pytest.approx(0)
    assert convert(0, 'C', 'K') == pytest.approx(273.15)
    for text in ['-273.15C', '-459.67F', '0K']:
        value, unit = parse_reading(text)
        assert convert(value, unit, 'K') == pytest.approx(0, abs=1e-10)
    for text in ['-273.16C', '-459.68F', '-.01K']:
        with pytest.raises(ValueError):
            parse_reading(text)


def test_invalid_numbers_and_units():
    for value in [True, '1', float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            convert(value, 'C', 'F')
    with pytest.raises(ValueError):
        convert(-1, 'K', 'C')
    for unit in ['', ' C', 'CC', 'X', None]:
        with pytest.raises(ValueError):
            convert(0, unit, 'C')
        with pytest.raises(ValueError):
            convert(0, 'C', unit)


def test_mixed_units_stats_and_format():
    log = TemperatureLog()
    assert log.add('freezing', '32F') is None
    log.add('boiling', '373.15K')
    log.add('room', '20C')
    assert log.stats() == pytest.approx({'count': 3, 'min': 0, 'max': 100, 'mean': 40})
    assert log.stats('F') == pytest.approx({'count': 3, 'min': 32, 'max': 212, 'mean': 104})
    assert log.format('f') == 'count: 3\nmin: 32.0 F\nmax: 212.0 F\nmean: 104.0 F'


def test_replace_order_and_snapshots():
    log = TemperatureLog()
    log.add(' a ', '0C')
    log.add('b', '10C')
    log.add('a', '68F')
    assert log.readings() == [('a', 20.0), ('b', 10.0)]
    log.readings().clear()
    snapshot = log.stats()
    snapshot['count'] = 99
    assert log.stats()['count'] == 2
    assert log.stats()['mean'] == pytest.approx(15)


def test_empty_log_and_invalid_add_atomicity():
    log = TemperatureLog()
    assert log.stats() == {'count': 0, 'min': None, 'max': None, 'mean': None}
    assert log.format('k') == 'count: 0\nmin: -- K\nmax: -- K\nmean: -- K'
    for method in [log.stats, log.readings, log.format]:
        with pytest.raises(ValueError):
            method('bad')
    log.add('a', '5C')
    for label, text in [(' ', '0C'), (None, '0C'), ('a', 'bad'), ('b', '-1K')]:
        with pytest.raises(ValueError):
            log.add(label, text)
    assert log.readings() == [('a', 5.0)]


def test_negative_zero_and_fractional_mean():
    log = TemperatureLog()
    log.add('a', '-.01C')
    assert log.format() == 'count: 1\nmin: 0.0 C\nmax: 0.0 C\nmean: 0.0 C'
    log.add('b', '.02C')
    assert log.stats()['mean'] == pytest.approx(.005)
    assert log.readings('K')[0][1] == pytest.approx(273.14)
