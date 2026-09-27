"""Integer-cent discount calculations for shopping carts."""


def percentage_discount(subtotal, percent, minimum=0):
    """Return a floored percentage saving when the minimum is reached."""
    for value in (subtotal, percent, minimum):
        if type(value) is not int:
            raise ValueError("rule arguments must be integers")
    if subtotal < 0 or minimum < 0 or not 0 <= percent <= 100:
        raise ValueError("invalid rule arguments")
    if subtotal < minimum:
        return 0
    return subtotal * percent // 100


def best_discount(subtotal, rules):
    """Validate every (percent, minimum) rule and choose one saving."""
    if type(subtotal) is not int or subtotal < 0:
        raise ValueError("invalid subtotal")
    savings = [percentage_discount(subtotal, percent, minimum)
               for percent, minimum in rules]
    return max(savings, default=0)
