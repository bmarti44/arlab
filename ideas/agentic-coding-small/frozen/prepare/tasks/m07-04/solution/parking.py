"""Deterministic best-fit parking allocation."""

KINDS = {"compact": 0, "standard": 1, "large": 2}


def _validate_kind(kind):
    if kind not in KINDS:
        raise ValueError("unknown vehicle or spot kind")
    return KINDS[kind]


def _choose_spot(spots, occupied, vehicle_kind):
    """Choose by capacity first, then by stable spot identifier."""
    needed = _validate_kind(vehicle_kind)
    candidates = []
    for spot_id, kind in spots.items():
        if spot_id in occupied:
            continue
        rank = KINDS[kind]
        if rank >= needed:
            candidates.append((rank, spot_id))
    if not candidates:
        return None
    return min(candidates)[1]


class ParkingLot:
    """Keep both plate and spot indexes for current assignments."""

    def __init__(self, spots: dict[str, str]):
        copied = {}
        for spot_id, kind in spots.items():
            if not spot_id:
                raise ValueError("empty spot identifier")
            _validate_kind(kind)
            copied[spot_id] = kind
        self._spots = copied
        self._plates = {}
        self._occupied = {}

    def park(self, plate: str, kind: str) -> str | None:
        if not plate:
            raise ValueError("empty plate")
        _validate_kind(kind)
        if plate in self._plates:
            raise ValueError("plate already parked")
        spot_id = _choose_spot(self._spots, self._occupied, kind)
        if spot_id is None:
            return None
        self._plates[plate] = spot_id
        self._occupied[spot_id] = plate
        return spot_id

    def leave(self, plate: str) -> str:
        if plate not in self._plates:
            raise KeyError(plate)
        spot_id = self._plates.pop(plate)
        del self._occupied[spot_id]
        return spot_id

    def locate(self, plate: str) -> str | None:
        return self._plates.get(plate)

    def snapshot(self) -> list[tuple[str, str, str | None]]:
        result = []
        for spot_id in sorted(self._spots):
            result.append((
                spot_id,
                self._spots[spot_id],
                self._occupied.get(spot_id),
            ))
        return result
