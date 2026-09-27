def parse_span(start, end):
    raise NotImplementedError


class BookingCalendar:
    def __init__(self, rooms):
        raise NotImplementedError

    def book(self, booking_id, room, start, end):
        raise NotImplementedError

    def cancel(self, booking_id):
        raise NotImplementedError

    def available(self, start, end):
        raise NotImplementedError

    def bookings(self, room=None):
        raise NotImplementedError
