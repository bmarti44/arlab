def next_occurrence(start, interval, after):
    """Find a strictly later occurrence without iterating through the series."""
    raise NotImplementedError()

class EventScheduler:
    """A deterministic scheduler driven explicitly by integer time limits."""

    def __init__(self):
        raise NotImplementedError()

    def add(self, event_id, when, payload=None, interval=None):
        """Insert an event only after all constraints have been checked."""
        raise NotImplementedError()

    def cancel(self, event_id):
        """Cancel both the next occurrence and any later recurrence."""
        raise NotImplementedError()

    def run_until(self, limit):
        """Drain due occurrences in time order, keeping stable tie order."""
        raise NotImplementedError()

    def pending(self):
        """Expose only the next occurrence of each live event."""
        raise NotImplementedError()
