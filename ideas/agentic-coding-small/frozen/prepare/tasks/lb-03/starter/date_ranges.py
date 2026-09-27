"""Utilities for inclusive ranges of calendar dates.

Endpoints use the exact YYYY-MM-DD spelling. All functions validate both
endpoints before performing range operations.
"""

import re
from datetime import date, timedelta


def _date(value):
    """Read one strict ISO calendar date."""
    if not isinstance(value, str):
        raise ValueError("date must be a string")
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
        raise ValueError("expected YYYY-MM-DD")
    return date.fromisoformat(value)


def _range(start, end):
    """Parse and validate an inclusive interval."""
    first = _date(start)
    last = _date(end)
    if first > last:
        raise ValueError("start follows end")
    return first, last


def dates_between(start, end):
    """Return every date, including both endpoints, as ISO strings.

    Calculating a day count also avoids stepping beyond date.max when
    the final endpoint is 9999-12-31.
    """
    first, last = _range(start, end)
    count = (last - first).days
    result = []
    for offset in range(count):
        current = first + timedelta(days=offset)
        result.append(current.isoformat())
    return result


def overlap(start_a, end_a, start_b, end_b):
    """Return the inclusive intersection, or None for disjoint ranges.

    A shared endpoint represents a one-day intersection.
    """
    first_a, last_a = _range(start_a, end_a)
    first_b, last_b = _range(start_b, end_b)
    first = max(first_a, first_b)
    last = min(last_a, last_b)
    if first >= last:
        return None
    return first.isoformat(), last.isoformat()
