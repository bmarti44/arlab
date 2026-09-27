def _validate_order(order_id, side, price, quantity, orders):
    """Validate the complete request before any matching takes place."""
    raise NotImplementedError()

class OrderBook:
    """One instrument's resting buy and sell limit orders."""

    def __init__(self):
        raise NotImplementedError()

    def _priority(self, order):
        raise NotImplementedError()

    def add(self, order_id, side, price, quantity):
        """Match an incoming order and return its executions."""
        raise NotImplementedError()

    def cancel(self, order_id):
        """Cancellation does not affect priority of other orders."""
        raise NotImplementedError()

    def snapshot(self, side):
        """Expose immutable records in a fresh, independently mutable list."""
        raise NotImplementedError()
