from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Customer:
    """An arrival and its uninterrupted service requirement."""
    customer_id: str
    arrival: int
    duration: int

def parse_customers(text):
    """Parse a small CSV-like input with exactly three unquoted fields.

    Parsing does not sort arrivals: input order is the queue tie breaker.
    The entire call fails if any nonblank line is invalid.
    """
    raise NotImplementedError()
