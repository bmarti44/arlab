def validate_path(path: str) -> None:
    """Reject absolute paths, traversal segments, and ambiguous separators."""
    if not path or "\\" in path:
        raise ValueError("invalid path")
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError("invalid path segment")


class SnapshotStore:
    """Own validated snapshots and monotonically allocate integer IDs."""

    def __init__(self):
        self._snapshots = {}
        self._next_id = 1

    def save(self, files: dict[str, str]) -> int:
        snapshot = {}
        for path, content in files.items():
            validate_path(path)
            if not isinstance(content, str):
                raise TypeError("file contents must be strings")
            snapshot[path] = content
        snapshot_id = self._next_id
        self._snapshots[snapshot_id] = snapshot
        self._next_id += 1
        return snapshot_id

    def load(self, snapshot_id: int) -> dict[str, str]:
        if type(snapshot_id) is not int or snapshot_id not in self._snapshots:
            raise KeyError(snapshot_id)
        return dict(self._snapshots[snapshot_id])
