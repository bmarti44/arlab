"""Named affine conversions through a common base for each dimension."""

import math
import re

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


class UnitRegistry:
    def __init__(self):
        self._units = {}

    def register(self, name, dimension, factor, offset=0.0):
        """Register base_value = value * factor + offset atomically."""
        if _NAME.fullmatch(name) is None:
            raise ValueError("invalid unit name")
        if not dimension:
            raise ValueError("empty dimension")
        if name in self._units:
            raise ValueError("duplicate unit")
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("factor must be positive and finite")
        if not math.isfinite(offset):
            raise ValueError("offset must be finite")
        self._units[name] = (dimension, float(factor), float(offset))
        return None

    def convert(self, value, source, target):
        """Convert only between units with equal dimension strings."""
        source_dimension, source_factor, source_offset = self._units[source]
        target_dimension, target_factor, target_offset = self._units[target]
        if source_dimension != target_dimension:
            raise ValueError("incompatible dimensions")
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        base_value = value * source_factor + source_offset
        result = (base_value - target_offset) / target_factor
        if not math.isfinite(result):
            raise ValueError("conversion overflow")
        return float(result)

    def units(self):
        """Return a fresh sorted list of case-sensitive unit names."""
        return sorted(self._units)
