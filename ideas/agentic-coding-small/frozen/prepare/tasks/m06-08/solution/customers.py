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
    result = []
    seen = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 3:
            raise ValueError('expected id,arrival,duration')
        customer_id, arrival_text, duration_text = fields
        if not customer_id or customer_id in seen:
            raise ValueError('customer ids must be nonempty and unique')
        for value in (arrival_text, duration_text):
            if re.fullmatch(r'[0-9]+', value) is None:
                raise ValueError('expected an unsigned integer')
        arrival = int(arrival_text)
        duration = int(duration_text)
        if duration == 0:
            raise ValueError('duration must be positive')
        result.append(Customer(customer_id, arrival, duration))
        seen.add(customer_id)
    return result
