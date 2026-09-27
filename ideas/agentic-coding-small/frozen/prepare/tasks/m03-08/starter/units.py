from fractions import Fraction
from math import isfinite
import re
_GROUPS = (('g', 1, 'g gram grams'), ('g', 1000, 'kg kilogram kilograms'), ('ml', 1, 'ml milliliter milliliters'), ('ml', 1000, 'l liter liters'), ('ml', 5, 'tsp teaspoon teaspoons'), ('ml', 15, 'tbsp tablespoon tablespoons'), ('ml', 240, 'cup cups'), ('each', 1, 'each piece pieces'))
_UNITS = {alias: (base, factor) for base, factor, aliases in _GROUPS for alias in aliases.split()}

def normalize_unit(unit: str) -> tuple[str, int]:
    """Map a known alias to a canonical dimension and integer factor."""
    raise NotImplementedError()

def parse_amount(value: int | float | str) -> Fraction:
    """Parse restricted human quantities without rounding their fractions."""
    raise NotImplementedError()
