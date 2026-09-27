from time_ranges import parse_time, overlap_minutes, overlaps, disjoint


def test_daytime_intersection_and_parser():
    assert overlap_minutes('09:00', '11:00', '10:00', '12:00') == 60
    assert parse_time('00:00') == 0
    assert parse_time('23:59') == 1439
    assert overlaps('09:00', '11:00', '10:00', '12:00') is True


def test_touching_and_empty_ranges():
    assert overlap_minutes('09:00', '10:00', '10:00', '11:00') == 0
    assert overlaps('09:00', '10:00', '10:00', '11:00') is False
    assert disjoint('09:00', '10:00', '10:00', '11:00') is True
    assert overlap_minutes('10:00', '10:00', '09:00', '11:00') == 0


def test_contained_and_identical_ranges():
    assert overlap_minutes('08:00', '14:00', '10:00', '10:01') == 1
    assert overlap_minutes('08:00', '14:00', '08:00', '14:00') == 360
    assert disjoint('08:00', '14:00', '08:00', '14:00') is False


def test_overnight_against_early_and_late_day():
    assert overlap_minutes('23:00', '02:00', '01:00', '03:00') == 60
    assert overlap_minutes('23:00', '02:00', '22:00', '23:30') == 30
    assert overlap_minutes('23:00', '02:00', '12:00', '13:00') == 0


def test_two_overnight_ranges():
    assert overlap_minutes('22:00', '03:00', '23:00', '02:00') == 180
    assert overlap_minutes('23:30', '00:30', '23:30', '00:30') == 60
    assert overlaps('23:30', '00:30', '00:00', '00:01') is True


def test_midnight_boundaries_and_symmetry():
    a = ('23:59', '00:01')
    b = ('00:00', '00:01')
    assert overlap_minutes(*a, *b) == 1
    assert overlap_minutes(*b, *a) == 1
    assert overlap_minutes('23:00', '00:00', '00:00', '01:00') == 0
    assert overlap_minutes('00:00', '00:00', *a) == 0
