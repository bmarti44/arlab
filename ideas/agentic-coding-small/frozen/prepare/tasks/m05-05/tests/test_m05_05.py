from datetime import datetime, timezone
import pytest
from weekly_cron import CronSchedule, parse_field


def test_parse_lists_ranges_and_duplicates():
    assert parse_field("01,3-5,4,1", 0, 9) == {1, 3, 4, 5}
    assert parse_field("*", 2, 4) == {2, 3, 4}
    assert parse_field("2-2", 0, 3) == {2}


def test_invalid_field_grammar_and_bounds():
    for field in ["", "1,", ",2", "1,,2", "*,2", "*/2", "-1", "+1", "2-1", "0-7", "7", "1 -2", " 1", "１", "1-2-3"]:
        with pytest.raises(ValueError):
            parse_field(field, 0, 6)


def test_expression_validation_and_matching():
    schedule = CronSchedule("  5,10-12   8  0-4  ")
    assert schedule.matches(datetime(2024, 1, 1, 8, 11, 59, 99))
    assert not schedule.matches(datetime(2024, 1, 6, 8, 11))
    assert not schedule.matches(datetime(2024, 1, 1, 9, 11))
    for expr in ["", "* *", "* * * *", "60 * *", "0 24 0", "0 0 7"]:
        with pytest.raises(ValueError):
            CronSchedule(expr)


def test_strictly_later_minute():
    schedule = CronSchedule("* * *")
    assert schedule.next_after(datetime(2024, 1, 1, 12, 5)) == datetime(2024, 1, 1, 12, 6)
    assert schedule.next_after(datetime(2024, 1, 1, 12, 5, 59, 999999)) == datetime(2024, 1, 1, 12, 6)


def test_weekly_wrap_and_weekday_numbering():
    schedule = CronSchedule("15 9 0")
    assert schedule.next_after(datetime(2024, 1, 1, 9, 15)) == datetime(2024, 1, 8, 9, 15)
    assert CronSchedule("0 0 6").next_after(datetime(2024, 1, 6, 23, 59)) == datetime(2024, 1, 7)


def test_calendar_rollovers():
    midnight = CronSchedule("0 0 *")
    assert midnight.next_after(datetime(2023, 12, 31, 23, 59, 30)) == datetime(2024, 1, 1)
    assert midnight.next_after(datetime(2024, 2, 28, 23, 59)) == datetime(2024, 2, 29)
    assert midnight.next_after(datetime(2024, 2, 29, 23, 59)) == datetime(2024, 3, 1)


def test_timezone_is_rejected():
    schedule = CronSchedule("* * *")
    aware = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        schedule.matches(aware)
    with pytest.raises(ValueError):
        schedule.next_after(aware)
