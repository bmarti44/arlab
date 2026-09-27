def validate_floor(floor: int, floors: int) -> None:
    """Validate a floor number against an already validated building size."""
    if type(floor) is not int:
        raise ValueError("floor must be an integer")
    if floor < 0 or floor >= floors:
        raise ValueError("floor outside building")


def new_state(floors: int, start: int = 0) -> dict:
    """Create an idle elevator with its doors closed."""
    if type(floors) is not int or floors <= 0:
        raise ValueError("invalid building size")
    validate_floor(start, floors)
    return {
        "floors": floors,
        "floor": start,
        "direction": 0,
        "doors_open": False,
        "requests": [],
    }
