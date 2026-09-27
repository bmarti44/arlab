def validate_path(path: str) -> None:
    """Reject absolute paths, traversal segments, and ambiguous separators."""
    raise NotImplementedError()

class SnapshotStore:
    """Own validated snapshots and monotonically allocate integer IDs."""

    def __init__(self):
        raise NotImplementedError()

    def save(self, files: dict[str, str]) -> int:
        raise NotImplementedError()

    def load(self, snapshot_id: int) -> dict[str, str]:
        raise NotImplementedError()
