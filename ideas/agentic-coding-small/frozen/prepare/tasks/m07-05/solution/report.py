"""Grouped integer-cent reporting with deterministic column alignment."""


def _money(cents):
    sign = "-" if cents < 0 else ""
    whole, fraction = divmod(abs(cents), 100)
    return f"{sign}{whole}.{fraction:02d}"


def _validate_row(group, item, quantity, unit_cents):
    for label in (group, item):
        if not label or any(char in label for char in "|\n\r"):
            raise ValueError("invalid label")
    if not isinstance(quantity, int) or quantity < 0:
        raise ValueError("invalid quantity")
    if not isinstance(unit_cents, int):
        raise ValueError("invalid price")


class SalesReport:
    """Collect rows in first-seen group order."""

    def __init__(self):
        self._groups = {}

    def add(self, group: str, item: str, quantity: int, unit_cents: int) -> None:
        _validate_row(group, item, quantity, unit_cents)
        if group not in self._groups:
            self._groups[group] = []
        self._groups[group].append((item, quantity, unit_cents))

    def render(self) -> str:
        table = [["Group", "Item", "Qty", "Amount"]]
        total_quantity = 0
        total_cents = 0
        for group, entries in self._groups.items():
            group_quantity = 0
            group_cents = 0
            for item, quantity, price in entries:
                cents = quantity * price
                table.append([
                    group,
                    item,
                    str(quantity),
                    _money(cents),
                ])
                group_quantity += quantity
                group_cents += cents
            table.append([
                group,
                "SUBTOTAL",
                str(group_quantity),
                _money(group_cents),
            ])
            total_quantity += group_quantity
            total_cents += group_cents
        table.append(["TOTAL", "", str(total_quantity), _money(total_cents)])

        widths = [0, 0, 0, 0]
        for row in table:
            for column, value in enumerate(row):
                widths[column] = max(widths[column], len(value))

        lines = []
        for row in table:
            cells = []
            for column, value in enumerate(row):
                if column < 2:
                    cells.append(value.ljust(widths[column]))
                else:
                    cells.append(value.rjust(widths[column]))
            lines.append(" | ".join(cells))
        return "\n".join(lines)
