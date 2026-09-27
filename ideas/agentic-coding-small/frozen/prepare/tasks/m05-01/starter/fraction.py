from math import gcd

class Fraction:
    """An exact rational with a positive, reduced denominator."""

    def __init__(self, numerator: int, denominator: int=1):
        """Validate integers and establish the canonical representation."""
        raise NotImplementedError()

    def add(self, other: 'Fraction') -> 'Fraction':
        """Return the sum, leaving both operands unchanged."""
        raise NotImplementedError()

    def subtract(self, other: 'Fraction') -> 'Fraction':
        """Return the signed difference in canonical form."""
        raise NotImplementedError()

    def multiply(self, other: 'Fraction') -> 'Fraction':
        """Return the product in canonical form."""
        raise NotImplementedError()

    def as_tuple(self) -> tuple[int, int]:
        """Expose the canonical pair without exposing mutable storage."""
        raise NotImplementedError()

    def __str__(self) -> str:
        """Keep the denominator visible even when it is one."""
        raise NotImplementedError()
