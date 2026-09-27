import re

def seat_label(row: int, number: int) -> str:
    raise NotImplementedError()

def parse_seat(label: str, rows: int, seats_per_row: int) -> tuple[int, int]:
    raise NotImplementedError()

class SeatMap:

    def __init__(self, rows: int, seats_per_row: int):
        raise NotImplementedError()

    def reserve(self, labels, owner: str) -> list[str]:
        """Validate the complete selection before assigning any seats."""
        raise NotImplementedError()

    def release(self, owner: str) -> list[str]:
        raise NotImplementedError()

    def available(self, row: str) -> list[str]:
        raise NotImplementedError()

    def reserve_block(self, row: str, size: int, owner: str) -> list[str] | None:
        """Scan consecutive runs in numerical seat order."""
        raise NotImplementedError()

    def snapshot(self) -> dict[str, str]:
        raise NotImplementedError()
