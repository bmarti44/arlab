from fractions import Fraction
from math import isfinite
import re


_GROUPS = (
    ("g", 1, "g gram grams"),
    ("g", 1000, "kg kilogram kilograms"),
    ("ml", 1, "ml milliliter milliliters"),
    ("ml", 1000, "l liter liters"),
    ("ml", 5, "tsp teaspoon teaspoons"),
    ("ml", 15, "tbsp tablespoon tablespoons"),
    ("ml", 240, "cup cups"),
    ("each", 1, "each piece pieces"),
)
_UNITS = {alias: (base, factor)
          for base, factor, aliases in _GROUPS for alias in aliases.split()}


def normalize_unit(unit: str) -> tuple[str, int]:
    """Map a known alias to a canonical dimension and integer factor."""
    key = unit.strip().lower()
    if key not in _UNITS:
        raise ValueError("unknown unit")
    return _UNITS[key]


def parse_amount(value: int | float | str) -> Fraction:
    """Parse restricted human quantities without rounding their fractions."""
    if type(value) in (int, float):
        if value < 0 or not isfinite(value):
            raise ValueError("invalid amount")
        return Fraction(str(value))
    if not isinstance(value, str):
        raise ValueError("invalid amount type")
    text = value.strip()
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text):
        return Fraction(text)
    match = re.fullmatch(r"(?:([0-9]+) )?([0-9]+)/([0-9]+)", text)
    if match is None:
        raise ValueError("invalid quantity syntax")
    whole_text, numerator_text, denominator_text = match.groups()
    whole = int(whole_text or 0)
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    if denominator == 0:
        raise ValueError("zero denominator")
    fraction = Fraction(numerator, denominator)
    return Fraction(whole) + fraction
