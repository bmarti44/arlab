from customers import Customer


def simulate(customers: list[Customer], tellers: int) -> list[dict]:
    """Assign a FIFO queue to the earliest available teller.

    A teller's next available time is sufficient state: each assignment
    reserves that teller through the new service's end.
    """
    if tellers <= 0:
        raise ValueError('at least one teller is required')
    free_at = [0] * tellers
    records = []
    for customer in sorted(customers, key=lambda item: item.arrival):
        # All already-idle tellers tie at this customer's arrival time.
        teller = min(
            range(tellers),
            key=lambda index: (max(free_at[index], customer.arrival), index),
        )
        start = max(customer.arrival, free_at[teller])
        end = start + customer.duration
        free_at[teller] = end
        records.append({
            'customer_id': customer.customer_id,
            'teller': teller,
            'arrival': customer.arrival,
            'start': start,
            'end': end,
            'wait': start - customer.arrival,
        })
    return records


def summarize(records, tellers):
    """Aggregate completed simulated records without changing them."""
    if tellers <= 0:
        raise ValueError('at least one teller is required')
    busy = [0] * tellers
    total_wait = 0
    max_wait = 0
    last_end = 0
    for record in records:
        busy[record['teller']] += record['end'] - record['start']
        total_wait += record['wait']
        max_wait = max(max_wait, record['wait'])
        last_end = max(last_end, record['end'])
    return {
        'served': len(records),
        'total_wait': total_wait,
        'max_wait': max_wait,
        'last_end': last_end,
        'busy': busy,
    }
