from elevator_state import validate_floor

def _choose_direction(floor: int, direction: int, pending: set[int]) -> int:
    """Keep a travel sweep until no requested floor remains ahead."""
    raise NotImplementedError()

def step(state: dict, new_requests=()) -> dict:
    """Advance one tick using a private request set and a fresh state dict."""
    raise NotImplementedError()
