class RateTable:
    """A graduated tariff whose final tier has no upper limit."""

    def __init__(self, tiers):
        copied = list(tiers)
        if not copied:
            raise ValueError('at least one tier is required')
        previous = 0
        normalized = []
        for index, (bound, price) in enumerate(copied):
            if price < 0:
                raise ValueError('negative price')
            is_last = index == len(copied) - 1
            if bound is None:
                if not is_last:
                    raise ValueError('unlimited tier must be last')
            else:
                if is_last:
                    raise ValueError('missing unlimited tier')
                if bound <= previous:
                    raise ValueError('bounds must increase from zero')
                previous = bound
            normalized.append((bound, price))
        self._tiers = tuple(normalized)

    def breakdown(self, quantity):
        if quantity < 0:
            raise ValueError('negative quantity')
        result = []
        remaining = quantity
        lower_bound = 0
        for upper_bound, unit_cents in self._tiers:
            if remaining == 0:
                break
            if upper_bound is None:
                units = remaining
            else:
                width = upper_bound - lower_bound
                units = min(remaining, width)
            if units > 0:
                result.append((units, unit_cents, units * unit_cents))
                remaining -= units
            if upper_bound is not None:
                lower_bound = upper_bound
        return result

    def quote(self, quantity):
        total = 0
        for units, unit_cents, subtotal in self.breakdown(quantity):
            total += subtotal
        return total


def format_money(cents):
    """Format exact cents without converting through floating point."""
    if cents < 0:
        raise ValueError('negative money')
    dollars, remainder = divmod(cents, 100)
    return f'${dollars}.{remainder:02d}'


def format_invoice(table, items):
    """Apply the full tariff independently to every invoice item."""
    lines = []
    total = 0
    for label, quantity in items:
        amount = table.quote(quantity)
        lines.append(f'{label}: {quantity} units = {format_money(amount)}')
        total += amount
    lines.append(f'Total: {format_money(total)}')
    return '\n'.join(lines)
