"""Shopping cart with immutable unit prices and integer-cent checkout."""

from discount_rules import best_discount


def _integer(value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid integer amount")


class Cart:
    def __init__(self):
        self._items = {}

    def add(self, sku, unit_price, quantity=1):
        """Add units; an existing SKU must retain its original price."""
        _integer(unit_price, 0)
        _integer(quantity, 1)
        if sku in self._items:
            old_price, old_quantity = self._items[sku]
            if unit_price != old_price:
                raise ValueError("price differs for existing SKU")
        self._items[sku] = (unit_price, quantity)
        return None

    def remove(self, sku, quantity=1):
        """Remove exactly the requested units or leave the cart unchanged."""
        _integer(quantity, 1)
        price, current = self._items[sku]
        if quantity > current:
            raise ValueError("insufficient quantity")
        if quantity == current:
            del self._items[sku]
        else:
            self._items[sku] = (price, current - quantity)
        return None

    def items(self):
        """Return a detached mapping to (unit price, quantity) tuples."""
        return dict(self._items)

    def subtotal(self):
        """Calculate cents from all current line items."""
        return sum(price * quantity for price, quantity in self._items.values())

    def checkout(self, rules=(), shipping=0, free_shipping_at=None):
        """Quote without mutating items; shipping eligibility is pre-discount."""
        _integer(shipping, 0)
        if free_shipping_at is not None:
            _integer(free_shipping_at, 0)
        subtotal = self.subtotal()
        discount = best_discount(subtotal, rules)
        charged_shipping = shipping
        if not self._items:
            charged_shipping = 0
        elif free_shipping_at is not None and subtotal - discount >= free_shipping_at:
            charged_shipping = 0
        total = subtotal - discount + charged_shipping
        return {
            "subtotal": subtotal,
            "discount": discount,
            "shipping": charged_shipping,
            "total": total,
        }
