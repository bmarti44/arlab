import re
from datetime import date, timedelta


def parse_date(text: str) -> date:
    if not isinstance(text, str):
        raise ValueError("date must be text")
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", text):
        raise ValueError("expected YYYY-MM-DD")
    return date.fromisoformat(text)


def _identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("empty identifier")
    return value.strip()


class LoanTracker:
    """Track one active borrower and due date per book."""

    def __init__(self):
        self._loans = {}

    def checkout(self, book: str, borrower: str, on: str, days: int = 14) -> str:
        """Validate the entire request before installing a loan."""
        book = _identifier(book)
        borrower = _identifier(borrower)
        started = parse_date(on)
        if type(days) is not int or days <= 0:
            raise ValueError("days must be positive")
        if book in self._loans:
            raise ValueError("book is already on loan")
        due = started + timedelta(days=days)
        self._loans[book] = (borrower, due)
        return due.isoformat()

    def return_book(self, book: str) -> str:
        """Remove exactly one active loan and return its borrower."""
        book = _identifier(book)
        borrower, due = self._loans.pop(book)
        return borrower

    def renew(self, book: str, on: str, days: int = 7) -> str:
        """Extend the due date rather than starting a new loan period."""
        book = _identifier(book)
        today = parse_date(on)
        if type(days) is not int or days <= 0:
            raise ValueError("days must be positive")
        borrower, due = self._loans[book]
        if today > due:
            raise ValueError("overdue loans cannot be renewed")
        due = due + timedelta(days=days)
        self._loans[book] = (borrower, due)
        return due.isoformat()

    def loans(self) -> list[tuple[str, str, str]]:
        result = []
        for book in sorted(self._loans):
            borrower, due = self._loans[book]
            result.append((book, borrower, due.isoformat()))
        return result

    def overdue(self, on: str) -> list[tuple[str, str, str]]:
        """Due today is still on time; sort oldest due dates first."""
        today = parse_date(on)
        result = []
        for book, (borrower, due) in self._loans.items():
            if due < today:
                result.append((book, borrower, due.isoformat()))
        result.sort(key=lambda record: (record[2], record[0]))
        return result
