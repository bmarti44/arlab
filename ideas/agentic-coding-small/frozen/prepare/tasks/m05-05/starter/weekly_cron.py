from datetime import datetime, timedelta
import re
_NUMBER = re.compile('[0-9]+')
_RANGE = re.compile('([0-9]+)-([0-9]+)')

def parse_field(text: str, minimum: int, maximum: int) -> set[int]:
    """Expand one wildcard, list, or inclusive-range field.

    Full matches deliberately reject signs, embedded spaces and partial
    numbers. Validating endpoints also validates every expanded range value.
    """
    raise NotImplementedError()

def _require_naive(moment: datetime) -> None:
    """Reject timezone-bearing values instead of silently dropping the zone."""
    raise NotImplementedError()

class CronSchedule:
    """A weekly schedule specified by minute, hour and Monday-based weekday."""

    def __init__(self, expression: str):
        raise NotImplementedError()

    def matches(self, moment: datetime) -> bool:
        """Match calendar fields; seconds are not part of the expression."""
        raise NotImplementedError()

    def next_after(self, moment: datetime) -> datetime:
        """Search minute boundaries strictly after the supplied timestamp.

        Each field is nonempty, and there is no date-of-month restriction,
        so at most one week's worth of minute candidates is necessary.
        """
        raise NotImplementedError()
