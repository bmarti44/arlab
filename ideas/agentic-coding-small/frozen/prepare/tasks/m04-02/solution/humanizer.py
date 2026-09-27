from datetime import datetime


class Humanizer:
    """Describe instants relative to an explicit, reproducible reference."""

    UNITS = (
        ('week', 604800),
        ('day', 86400),
        ('hour', 3600),
        ('minute', 60),
        ('second', 1),
    )

    def __init__(self, now):
        if not isinstance(now, datetime):
            raise TypeError('now must be a datetime')
        self.now = now

    def seconds_from(self, moment):
        """Return signed whole seconds, discarding fractional seconds."""
        if not isinstance(moment, datetime):
            raise TypeError('moment must be a datetime')
        now_aware = self.now.utcoffset() is not None
        moment_aware = moment.utcoffset() is not None
        if now_aware != moment_aware:
            raise ValueError('cannot mix aware and naive datetimes')
        difference = moment - self.now
        return int(difference.total_seconds())

    def describe(self, moment):
        seconds = self.seconds_from(moment)
        if seconds == 0:
            return 'just now'
        magnitude = abs(seconds)
        for unit, size in self.UNITS:
            if magnitude >= size:
                count = magnitude // size
                suffix = '' if count == 1 else 's'
                phrase = f'{count} {unit}{suffix}'
                if seconds < 0:
                    return f'{phrase} ago'
                return f'in {phrase}'

    def describe_many(self, moments):
        """Consume an iterable once, preserving its order."""
        descriptions = []
        for moment in moments:
            descriptions.append(self.describe(moment))
        return descriptions
