from snapshot_store import SnapshotStore, validate_path

class History:
    """Track snapshots and ancestry separately from uncommitted edits."""

    def __init__(self):
        raise NotImplementedError()

    def write(self, path: str, content: str) -> None:
        raise NotImplementedError()

    def delete(self, path: str) -> bool:
        raise NotImplementedError()

    def files(self) -> dict[str, str]:
        raise NotImplementedError()

    def commit(self, message: str) -> int:
        """Validate metadata before allocating a snapshot identifier."""
        raise NotImplementedError()

    def checkout(self, commit_id: int) -> None:
        """Load before changing head so failed lookups are atomic."""
        raise NotImplementedError()

    def log(self, limit: int | None=None) -> list[dict]:
        """Walk the current branch, which need not include every stored ID."""
        raise NotImplementedError()
