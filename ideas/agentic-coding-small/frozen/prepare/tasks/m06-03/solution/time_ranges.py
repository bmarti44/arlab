"""Minute-resolution overlap on a repeating, timezone-free day."""

DAY_MINUTES = 24 * 60


def parse_time(text):
    """Convert a valid HH:MM time to minutes since midnight."""
    hours, minutes = text.split(':')
    return int(hours) * 60 + int(minutes)


def _segments(start, end):
    """Represent a half-open range as ordinary intervals in one day.

    Equal endpoints describe an empty interval, not a full day.
    A decreasing pair crosses midnight and therefore needs two pieces.
    """
    if start == end:
        return []
    if start < end:
        return [(start, end)]
    return [(start, DAY_MINUTES), (0, end)]


def overlap_minutes(start_a, end_a, start_b, end_b):
    """Return the number of minutes common to two daily ranges.

    End times are excluded, so touching intervals have no overlap.
    Times carry neither dates nor timezones; overnight ranges refer
    to the end and beginning of the same repeating 24-hour cycle.
    """
    a_start = parse_time(start_a)
    a_end = parse_time(end_a)
    b_start = parse_time(start_b)
    b_end = parse_time(end_b)
    segments_a = _segments(a_start, a_end)
    segments_b = _segments(b_start, b_end)

    total = 0
    for low_a, high_a in segments_a:
        for low_b, high_b in segments_b:
            left = max(low_a, low_b)
            right = min(high_a, high_b)
            total += max(0, right - left)
    return total


def overlaps(start_a, end_a, start_b, end_b):
    """Return whether the two ranges share at least one minute."""
    minutes = overlap_minutes(start_a, end_a, start_b, end_b)
    return minutes > 0


def disjoint(start_a, end_a, start_b, end_b):
    """Return the logical opposite of overlaps."""
    return not overlaps(start_a, end_a, start_b, end_b)
