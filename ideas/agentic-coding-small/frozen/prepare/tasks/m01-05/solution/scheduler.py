def next_occurrence(start, interval, after):
    """Find a strictly later occurrence without iterating through the series."""
    if interval <= 0:
        raise ValueError("interval must be positive")
    if after < start:
        return start
    steps = (after - start) // interval + 1
    return start + steps * interval


class EventScheduler:
    """A deterministic scheduler driven explicitly by integer time limits."""

    def __init__(self):
        self.now = 0
        self._events = {}
        self._sequence = 0

    def add(self, event_id, when, payload=None, interval=None):
        """Insert an event only after all constraints have been checked."""
        if event_id in self._events:
            raise ValueError("duplicate event ID")
        if when < self.now:
            raise ValueError("event is in the past")
        if interval is not None and interval <= 0:
            raise ValueError("interval must be positive")
        self._events[event_id] = {
            "when": when,
            "order": self._sequence,
            "payload": payload,
            "interval": interval,
        }
        self._sequence += 1

    def cancel(self, event_id):
        """Cancel both the next occurrence and any later recurrence."""
        if event_id not in self._events:
            return False
        del self._events[event_id]
        return True

    def run_until(self, limit):
        """Drain due occurrences in time order, keeping stable tie order."""
        if limit < self.now:
            raise ValueError("cannot move time backwards")
        emitted = []
        while self._events:
            event_id = min(
                self._events,
                key=lambda key: (self._events[key]["when"],
                                 self._events[key]["order"]),
            )
            event = self._events[event_id]
            when = event["when"]
            if when > limit:
                break
            emitted.append((when, event_id, event["payload"]))
            interval = event["interval"]
            if interval is None:
                del self._events[event_id]
            else:
                event["when"] = next_occurrence(when, interval, when)
        self.now = limit
        return emitted

    def pending(self):
        """Expose only the next occurrence of each live event."""
        ordered = sorted(
            self._events.items(),
            key=lambda pair: (pair[1]["when"], pair[1]["order"]),
        )
        return [(event["when"], event_id) for event_id, event in ordered]
