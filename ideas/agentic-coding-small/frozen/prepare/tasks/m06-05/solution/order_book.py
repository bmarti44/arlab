def _validate_order(order_id, side, price, quantity, orders):
    """Validate the complete request before any matching takes place."""
    if not order_id or any(order['id'] == order_id for order in orders):
        raise ValueError('order id must be nonempty and inactive')
    if side not in ('buy', 'sell'):
        raise ValueError('unknown side')
    if price <= 0 or quantity <= 0:
        raise ValueError('price and quantity must be positive')


class OrderBook:
    """One instrument's resting buy and sell limit orders."""

    def __init__(self):
        self._orders = []
        self._sequence = 0

    def _priority(self, order):
        price = order['price']
        if order['side'] == 'buy':
            price = -price
        return price, order['sequence']

    def add(self, order_id, side, price, quantity):
        """Match an incoming order and return its executions."""
        _validate_order(order_id, side, price, quantity, self._orders)
        opposite = 'sell' if side == 'buy' else 'buy'
        candidates = sorted(
            (order for order in self._orders if order['side'] == opposite),
            key=self._priority,
        )
        executions = []
        remaining = quantity
        for resting in candidates:
            if side == 'buy' and resting['price'] > price:
                break
            if side == 'sell' and resting['price'] < price:
                break
            filled = min(remaining, resting['quantity'])
            if side == 'buy':
                buy_id, sell_id = order_id, resting['id']
            else:
                buy_id, sell_id = resting['id'], order_id
            executions.append((buy_id, sell_id, resting['price'], filled))
            remaining -= filled
            resting['quantity'] -= filled
            if resting['quantity'] == 0:
                self._orders.remove(resting)
            if remaining == 0:
                break

        if remaining:
            self._orders.append({
                'id': order_id,
                'side': side,
                'price': price,
                'quantity': remaining,
                'sequence': self._sequence,
            })
        self._sequence += 1
        return executions

    def cancel(self, order_id):
        """Cancellation does not affect priority of other orders."""
        for order in self._orders:
            if order['id'] == order_id:
                self._orders.remove(order)
                return True
        return False

    def snapshot(self, side):
        """Expose immutable records in a fresh, independently mutable list."""
        if side not in ('buy', 'sell'):
            raise ValueError('unknown side')
        orders = sorted(
            (order for order in self._orders if order['side'] == side),
            key=self._priority,
        )
        return [(order['id'], order['price'], order['quantity'])
                for order in orders]
