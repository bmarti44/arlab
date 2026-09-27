import math
from temperature import convert, parse_reading


class TemperatureLog:
    """Retain original units to avoid unnecessary conversion round trips."""

    def __init__(self):
        self._readings = {}

    def add(self, label: str, text: str) -> None:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label must not be empty")
        reading = parse_reading(text)
        self._readings[label.strip()] = reading

    def readings(self, unit: str = "C") -> list[tuple[str, float]]:
        convert(0, "K", unit)
        result = []
        for label, (value, source) in self._readings.items():
            result.append((label, convert(value, source, unit)))
        return result

    def stats(self, unit: str = "C") -> dict:
        values = [value for label, value in self.readings(unit)]
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": math.fsum(values) / len(values),
        }

    def format(self, unit: str = "C") -> str:
        stats = self.stats(unit)
        lines = [f"count: {stats['count']}"]
        for name in ("min", "max", "mean"):
            value = stats[name]
            rendered = "--" if value is None else f"{value:.1f}"
            if rendered == "-0.0":
                rendered = "0.0"
            lines.append(f"{name}: {rendered} {unit.upper()}")
        return "\n".join(lines)
