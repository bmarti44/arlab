from elevator_state import validate_floor


def _choose_direction(floor: int, direction: int, pending: set[int]) -> int:
    """Keep a travel sweep until no requested floor remains ahead."""
    if direction == 1:
        if any(target > floor for target in pending):
            return 1
        return -1
    if direction == -1:
        if any(target < floor for target in pending):
            return -1
        return 1
    nearest = min(pending, key=lambda target: (abs(target - floor), target))
    return 1 if nearest > floor else -1


def step(state: dict, new_requests=()) -> dict:
    """Advance one tick using a private request set and a fresh state dict."""
    # Validate additions before constructing the next public state.
    pending = set(state["requests"])
    for target in new_requests:
        validate_floor(target, state["floors"])
        pending.add(target)
    result = {
        "floors": state["floors"],
        "floor": state["floor"],
        "direction": state["direction"],
        "doors_open": state["doors_open"],
        "requests": [],
    }
    floor = result["floor"]
    # Door closing takes a complete tick, including at the final stop.
    if result["doors_open"]:
        result["doors_open"] = False
        if not pending:
            result["direction"] = 0
        result["requests"] = sorted(pending)
        return result
    # A request here has precedence over selecting a travel direction.
    if floor in pending:
        pending.remove(floor)
        result["doors_open"] = True
        if not pending:
            result["direction"] = 0
        result["requests"] = sorted(pending)
        return result
    if not pending:
        result["direction"] = 0
        return result
    # A movement tick can also serve its arrival floor.
    direction = _choose_direction(floor, result["direction"], pending)
    result["direction"] = direction
    result["floor"] = floor + direction
    if result["floor"] in pending:
        pending.remove(result["floor"])
        result["doors_open"] = True
    if not pending:
        result["direction"] = 0
    result["requests"] = sorted(pending)
    return result
