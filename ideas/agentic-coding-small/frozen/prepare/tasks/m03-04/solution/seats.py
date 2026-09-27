import re


def seat_label(row: int, number: int) -> str:
    if type(row) is not int or not 0 <= row < 26:
        raise ValueError("invalid row")
    if type(number) is not int or number < 1:
        raise ValueError("invalid number")
    return f"{chr(65 + row)}{number}"


def parse_seat(label: str, rows: int, seats_per_row: int) -> tuple[int, int]:
    match = re.fullmatch(r"([A-Z])([1-9][0-9]*)", label)
    if match is None:
        raise ValueError("invalid label")
    row = ord(match[1]) - 65
    number = int(match[2])
    if row >= rows or number > seats_per_row:
        raise ValueError("seat out of bounds")
    return row, number


class SeatMap:
    def __init__(self, rows: int, seats_per_row: int):
        if type(rows) is not int or not 1 <= rows <= 26:
            raise ValueError("invalid row count")
        if type(seats_per_row) is not int or seats_per_row < 1:
            raise ValueError("invalid row width")
        self.rows = rows
        self.width = seats_per_row
        self._occupied = {}

    def reserve(self, labels, owner: str) -> list[str]:
        """Validate the complete selection before assigning any seats."""
        if not isinstance(owner, str) or not owner:
            raise ValueError("invalid owner")
        positions = sorted({parse_seat(s, self.rows, self.width) for s in labels})
        if any(pos in self._occupied for pos in positions):
            raise ValueError("seat occupied")
        for pos in positions:
            self._occupied[pos] = owner
        return [seat_label(*pos) for pos in positions]

    def release(self, owner: str) -> list[str]:
        positions = sorted(pos for pos, holder in self._occupied.items() if holder == owner)
        for pos in positions:
            del self._occupied[pos]
        return [seat_label(*pos) for pos in positions]

    def available(self, row: str) -> list[str]:
        if len(row) != 1:
            raise ValueError("invalid row")
        row_index, _ = parse_seat(row + "1", self.rows, self.width)
        return [seat_label(row_index, n) for n in range(1, self.width + 1)
                if (row_index, n) not in self._occupied]

    def reserve_block(self, row: str, size: int, owner: str) -> list[str] | None:
        """Scan consecutive runs in numerical seat order."""
        free = set(self.available(row))
        if type(size) is not int or size < 1:
            raise ValueError("invalid block size")
        if not isinstance(owner, str) or not owner:
            raise ValueError("invalid owner")
        for first in range(1, self.width - size + 2):
            labels = [f"{row}{n}" for n in range(first, first + size)]
            if all(label in free for label in labels):
                return self.reserve(labels, owner)
        return None

    def snapshot(self) -> dict[str, str]:
        return {seat_label(*pos): self._occupied[pos] for pos in sorted(self._occupied)}
