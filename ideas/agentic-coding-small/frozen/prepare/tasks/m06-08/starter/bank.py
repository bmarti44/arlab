from customers import Customer

def simulate(customers: list[Customer], tellers: int) -> list[dict]:
    """Assign a FIFO queue to the earliest available teller.

    A teller's next available time is sufficient state: each assignment
    reserves that teller through the new service's end.
    """
    raise NotImplementedError()

def summarize(records, tellers):
    """Aggregate completed simulated records without changing them."""
    raise NotImplementedError()
