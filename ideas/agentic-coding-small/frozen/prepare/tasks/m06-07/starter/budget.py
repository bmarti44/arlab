import re
from datetime import date as CalendarDate

def _parse_date(text):
    """Reject alternate ISO forms before checking calendar validity."""
    raise NotImplementedError()

def _category(text):
    raise NotImplementedError()

class BudgetTracker:
    """Track signed spending and recurring category limits in cents."""

    def __init__(self):
        raise NotImplementedError()

    def add(self, entry_id, date, category, amount):
        """Validate fully before adding an entry to the ledger."""
        raise NotImplementedError()

    def remove(self, entry_id):
        raise NotImplementedError()

    def set_limit(self, category, cents):
        raise NotImplementedError()

    def report(self, month):
        """Aggregate only the selected calendar month.

        Limits contribute categories even in months without entries.
        A category remains visible when expenses and refunds cancel.
        """
        raise NotImplementedError()
