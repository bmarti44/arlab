import math
import re


def convert(value: int | float, from_unit: str, to_unit: str) -> float:
    units = []
    for unit in (from_unit, to_unit):
        if not isinstance(unit, str) or unit.upper() not in ("C", "F", "K"):
            raise ValueError("invalid temperature unit")
        units.append(unit.upper())
    source, target = units
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("temperature must be finite")
    lower = {"C": -273.15, "F": -459.67, "K": 0.0}
    if value < lower[source]:
        raise ValueError("below absolute zero")
    if source == target:
        return float(value)
    if source == "F":
        celsius = (value - 32) * 5 / 9
    elif source == "K":
        celsius = value - 273.15
    else:
        celsius = float(value)
    if target == "F":
        return celsius * 9 / 5 + 32
    if target == "K":
        return celsius + 273.15
    return celsius


def parse_reading(text: str) -> tuple[float, str]:
    if not isinstance(text, str):
        raise ValueError("reading must be text")
    pattern = r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))[ \t]*([CFKcfk])"
    match = re.fullmatch(pattern, text.strip())
    if match is None:
        raise ValueError("invalid reading")
    value = float(match.group(1))
    unit = match.group(2).upper()
    convert(value, unit, unit)
    return value, unit
