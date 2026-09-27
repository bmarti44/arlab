"""Validate connections on an explicit day-and-minute timeline."""
import re


def parse_time(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)@([0-9]{2}):([0-9]{2})", value)
    if match is None:
        raise ValueError("invalid time syntax")
    day, hour, minute = map(int, match.groups())
    if hour > 23 or minute > 59:
        raise ValueError("time out of range")
    return day * 1440 + hour * 60 + minute


class Itinerary:
    """Store valid individual legs and report cross-leg issues separately."""

    def __init__(self, min_connection: int = 30):
        if not isinstance(min_connection, int) or min_connection < 0:
            raise ValueError("invalid minimum connection")
        self._minimum = min_connection
        self._legs = []

    def add(self, origin: str, destination: str, departure: str, arrival: str) -> None:
        if not origin or not destination:
            raise ValueError("empty location")
        start = parse_time(departure)
        finish = parse_time(arrival)
        if finish <= start:
            raise ValueError("arrival must follow departure")
        leg = (origin, destination, start, finish)
        self._legs.append(leg)

    def validate(self) -> list[tuple[int, str]]:
        issues = []
        for index in range(1, len(self._legs)):
            previous = self._legs[index - 1]
            current = self._legs[index]
            previous_destination = previous[1]
            current_origin = current[0]
            if current_origin != previous_destination:
                issues.append((index, "location"))

            previous_arrival = previous[3]
            current_departure = current[2]
            gap = current_departure - previous_arrival
            if gap < 0:
                issues.append((index, "overlap"))
            elif gap < self._minimum:
                issues.append((index, "connection"))
        return issues

    def elapsed(self) -> int:
        if not self._legs:
            return 0
        first_departure = self._legs[0][2]
        last_arrival = self._legs[-1][3]
        return last_arrival - first_departure

    def legs(self) -> list[tuple[str, str, int, int]]:
        # The list is fresh and its tuple entries are immutable.
        result = []
        for origin, destination, departure, arrival in self._legs:
            result.append((
                origin,
                destination,
                departure,
                arrival,
            ))
        return result
