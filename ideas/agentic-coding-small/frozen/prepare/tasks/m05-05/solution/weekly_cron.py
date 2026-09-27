from datetime import datetime, timedelta
import re


_NUMBER = re.compile(r"[0-9]+")
_RANGE = re.compile(r"([0-9]+)-([0-9]+)")


def parse_field(text: str, minimum: int, maximum: int) -> set[int]:
    """Expand one wildcard, list, or inclusive-range field.

    Full matches deliberately reject signs, embedded spaces and partial
    numbers. Validating endpoints also validates every expanded range value.
    """
    if text == "*":
        return set(range(minimum, maximum + 1))
    values = set()
    for part in text.split(","):
        if _NUMBER.fullmatch(part):
            start = end = int(part)
        else:
            match = _RANGE.fullmatch(part)
            if match is None:
                raise ValueError("invalid field component")
            start, end = map(int, match.groups())
        if start > end:
            raise ValueError("range is reversed")
        if start < minimum or end > maximum:
            raise ValueError("field value is outside bounds")
        values.update(range(start, end + 1))
    return values


def _require_naive(moment: datetime) -> None:
    """Reject timezone-bearing values instead of silently dropping the zone."""
    if moment.tzinfo is not None:
        raise ValueError("naive datetime required")


class CronSchedule:
    """A weekly schedule specified by minute, hour and Monday-based weekday."""

    def __init__(self, expression: str):
        fields = expression.split()
        if len(fields) != 3:
            raise ValueError("expected minute, hour and weekday")
        self._minutes = parse_field(fields[0], 0, 59)
        self._hours = parse_field(fields[1], 0, 23)
        self._weekdays = parse_field(fields[2], 0, 6)

    def matches(self, moment: datetime) -> bool:
        """Match calendar fields; seconds are not part of the expression."""
        _require_naive(moment)
        return (moment.minute in self._minutes
                and moment.hour in self._hours
                and moment.weekday() in self._weekdays)

    def next_after(self, moment: datetime) -> datetime:
        """Search minute boundaries strictly after the supplied timestamp.

        Each field is nonempty, and there is no date-of-month restriction,
        so at most one week's worth of minute candidates is necessary.
        """
        _require_naive(moment)
        candidate = moment.replace(second=0, microsecond=0)
        candidate += timedelta(minutes=1)
        for _ in range(7 * 24 * 60):
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        raise ValueError("schedule has no next run")
