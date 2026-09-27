def parse_time(value: str) -> int:
    raise NotImplementedError


class Itinerary:
    def __init__(self, min_connection: int = 30):
        raise NotImplementedError

    def add(self, origin: str, destination: str, departure: str, arrival: str) -> None:
        raise NotImplementedError

    def validate(self) -> list[tuple[int, str]]:
        raise NotImplementedError

    def elapsed(self) -> int:
        raise NotImplementedError

    def legs(self) -> list[tuple[str, str, int, int]]:
        raise NotImplementedError
