class ParkingLot:
    def __init__(self, spots: dict[str, str]):
        raise NotImplementedError

    def park(self, plate: str, kind: str) -> str | None:
        raise NotImplementedError

    def leave(self, plate: str) -> str:
        raise NotImplementedError

    def locate(self, plate: str) -> str | None:
        raise NotImplementedError

    def snapshot(self) -> list[tuple[str, str, str | None]]:
        raise NotImplementedError
