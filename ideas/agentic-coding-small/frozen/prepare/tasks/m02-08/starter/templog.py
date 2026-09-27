from temperature import convert, parse_reading


class TemperatureLog:
    def __init__(self):
        self._readings = {}

    def add(self, label: str, text: str) -> None:
        raise NotImplementedError

    def readings(self, unit: str = "C") -> list[tuple[str, float]]:
        raise NotImplementedError

    def stats(self, unit: str = "C") -> dict:
        raise NotImplementedError

    def format(self, unit: str = "C") -> str:
        raise NotImplementedError
