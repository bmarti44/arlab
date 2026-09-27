from math import gcd


class Fraction:
    """An exact rational with a positive, reduced denominator."""

    def __init__(self, numerator: int, denominator: int = 1):
        """Validate integers and establish the canonical representation."""
        if type(numerator) is not int or type(denominator) is not int:
            raise TypeError("fraction components must be integers")
        if denominator == 0:
            raise ValueError("denominator must not be zero")
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        divisor = gcd(numerator, denominator)
        self.numerator = numerator // divisor
        self.denominator = denominator // divisor

    def add(self, other: 'Fraction') -> 'Fraction':
        """Return the sum, leaving both operands unchanged."""
        numerator = (self.numerator * other.denominator
                     + other.numerator * self.denominator)
        denominator = self.denominator * other.denominator
        return Fraction(numerator, denominator)

    def subtract(self, other: 'Fraction') -> 'Fraction':
        """Return the signed difference in canonical form."""
        numerator = (self.numerator * other.denominator
                     - other.numerator * self.denominator)
        denominator = self.denominator * other.denominator
        return Fraction(numerator, denominator)

    def multiply(self, other: 'Fraction') -> 'Fraction':
        """Return the product in canonical form."""
        numerator = self.numerator * other.numerator
        denominator = self.denominator * other.denominator
        return Fraction(numerator, denominator)

    def as_tuple(self) -> tuple[int, int]:
        """Expose the canonical pair without exposing mutable storage."""
        return self.numerator, self.denominator

    def __str__(self) -> str:
        """Keep the denominator visible even when it is one."""
        return f"{self.numerator}/{self.denominator}"
