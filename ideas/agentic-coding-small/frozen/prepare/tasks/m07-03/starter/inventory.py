"""Purchase orders for a stock snapshot; inputs are never modified."""
import math


def _validate_item(item):
    required = ("sku", "on_hand", "reserved", "incoming", "daily_rate", "pack_size")
    if any(key not in item for key in required):
        raise ValueError("missing field")
    if not isinstance(item["sku"], str) or not item["sku"]:
        raise ValueError("invalid sku")
    for field in ("on_hand", "reserved", "incoming", "pack_size"):
        value = item[field]
        if not isinstance(value, int) or value < 0:
            raise ValueError("invalid stock quantity")
    if item["pack_size"] == 0:
        raise ValueError("pack size must be positive")
    rate = item["daily_rate"]
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0:
        raise ValueError("invalid rate")


def _available(item):
    # Reservations consume existing stock; incoming units are future supply.
    return max(0, item["on_hand"] - item["reserved"])


def _pack_order(shortage, pack_size):
    # Suppliers accept only whole packs.
    return (shortage // pack_size) * pack_size


def calculate_reorders(items, horizon_days):
    """Return positive purchase orders in the order of the input records.

    Each order contains sku, available, target, and order quantities.
    A target covers the given number of days at the item's daily rate.
    Reservations exceeding stock do not create negative available stock.
    """
    if not isinstance(horizon_days, int) or horizon_days < 0:
        raise ValueError("invalid horizon")
    orders = []
    seen = set()
    for item in items:
        _validate_item(item)
        sku = item["sku"]
        if sku in seen:
            raise ValueError("duplicate sku")
        seen.add(sku)
        available = _available(item)
        target = math.ceil(item["daily_rate"] * horizon_days)
        shortage = max(0, target - available)
        quantity = _pack_order(shortage, item["pack_size"])
        if quantity > 0:
            orders.append({
                "sku": sku,
                "available": available,
                "target": target,
                "order": quantity,
            })
    return orders
