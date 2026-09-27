from datetime import date


def parse_date(text: str) -> date:
    raise NotImplementedError


class LoanTracker:
    def __init__(self):
        self._loans = {}

    def checkout(self, book: str, borrower: str, on: str, days: int = 14) -> str:
        raise NotImplementedError

    def return_book(self, book: str) -> str:
        raise NotImplementedError

    def renew(self, book: str, on: str, days: int = 7) -> str:
        raise NotImplementedError

    def loans(self) -> list[tuple[str, str, str]]:
        raise NotImplementedError

    def overdue(self, on: str) -> list[tuple[str, str, str]]:
        raise NotImplementedError
