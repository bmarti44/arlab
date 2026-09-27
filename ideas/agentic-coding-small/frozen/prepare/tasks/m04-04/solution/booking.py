from datetime import date


def parse_span(start, end):
    """Validate a half-open interval with canonical ISO date endpoints."""
    parsed = []
    for value in (start, end):
        try:
            day = date.fromisoformat(value)
        except ValueError:
            raise ValueError('invalid ISO date') from None
        if day.isoformat() != value:
            raise ValueError('expected YYYY-MM-DD')
        parsed.append(day)
    first, last = parsed
    if first >= last:
        raise ValueError('checkout must follow checkin')
    return first, last


class BookingCalendar:
    """Maintain bookings with exclusive checkout dates."""

    def __init__(self, rooms):
        self._rooms = tuple(sorted(rooms))
        self._bookings = {}

    def book(self, booking_id, room, start, end):
        """Check every constraint before committing the new booking."""
        if booking_id in self._bookings:
            raise ValueError('duplicate booking ID')
        if room not in self._rooms:
            raise ValueError('unknown room')
        first, last = parse_span(start, end)
        for existing in self._bookings.values():
            other_room, other_start, other_end = existing
            if other_room != room:
                continue
            if first < other_end and other_start < last:
                raise ValueError('overlapping booking')
        self._bookings[booking_id] = (room, first, last)

    def cancel(self, booking_id):
        """Remove an existing reservation, freeing its ID and room interval."""
        if booking_id not in self._bookings:
            raise KeyError(booking_id)
        del self._bookings[booking_id]

    def available(self, start, end):
        """Return rooms free for the entire requested interval."""
        first, last = parse_span(start, end)
        occupied = set()
        for room, other_start, other_end in self._bookings.values():
            if first < other_end and other_start < last:
                occupied.add(room)
        return [room for room in self._rooms if room not in occupied]

    def bookings(self, room=None):
        """Build a detached, consistently ordered view of reservations."""
        if room is not None and room not in self._rooms:
            raise ValueError('unknown room')
        result = []
        for booking_id, record in self._bookings.items():
            name, first, last = record
            if room is not None and room != name:
                continue
            result.append((
                booking_id,
                name,
                first.isoformat(),
                last.isoformat(),
            ))
        result.sort(key=lambda item: (item[2], item[1], item[0]))
        return result
