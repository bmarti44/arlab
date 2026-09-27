import re
from datetime import date as CalendarDate


def _parse_date(text):
    """Reject alternate ISO forms before checking calendar validity."""
    if re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', text) is None:
        raise ValueError('expected YYYY-MM-DD')
    return CalendarDate.fromisoformat(text)


def _category(text):
    name = text.strip()
    if not name:
        raise ValueError('category cannot be empty')
    return name


class BudgetTracker:
    """Track signed spending and recurring category limits in cents."""

    def __init__(self):
        self._entries = {}
        self._limits = {}

    def add(self, entry_id, date, category, amount):
        """Validate fully before adding an entry to the ledger."""
        if not entry_id or entry_id in self._entries:
            raise ValueError('entry id must be nonempty and unused')
        parsed = _parse_date(date)
        name = _category(category)
        if amount == 0:
            raise ValueError('amount cannot be zero')
        self._entries[entry_id] = (parsed, name, amount)

    def remove(self, entry_id):
        if entry_id not in self._entries:
            return False
        del self._entries[entry_id]
        return True

    def set_limit(self, category, cents):
        name = _category(category)
        if cents < 0:
            raise ValueError('limit must be nonnegative')
        self._limits[name] = cents

    def report(self, month):
        """Aggregate only the selected calendar month.

        Limits contribute categories even in months without entries.
        A category remains visible when expenses and refunds cancel.
        """
        if re.fullmatch(r'[0-9]{4}-[0-9]{2}', month) is None:
            raise ValueError('expected YYYY-MM')
        first_day = _parse_date(month + '-01')
        spent = {name: 0 for name in self._limits}
        for entry_date, category, amount in self._entries.values():
            if (entry_date.year, entry_date.month) != (
                first_day.year, first_day.month
            ):
                continue
            spent[category] = spent.get(category, 0) + amount

        rows = []
        for category in sorted(spent):
            limit = self._limits.get(category)
            amount = spent[category]
            remaining = None if limit is None else limit - amount
            rows.append({
                'category': category,
                'spent': amount,
                'limit': limit,
                'remaining': remaining,
            })
        return {
            'month': month,
            'total': sum(spent.values()),
            'categories': rows,
        }
