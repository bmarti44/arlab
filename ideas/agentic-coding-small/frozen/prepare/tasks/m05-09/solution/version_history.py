from snapshot_store import SnapshotStore, validate_path


class History:
    """Track snapshots and ancestry separately from uncommitted edits."""

    def __init__(self):
        self._store = SnapshotStore()
        self._working = {}
        self._commits = {}
        self.head = None

    def write(self, path: str, content: str) -> None:
        validate_path(path)
        if not isinstance(content, str):
            raise TypeError("file contents must be strings")
        self._working[path] = content

    def delete(self, path: str) -> bool:
        validate_path(path)
        if path not in self._working:
            return False
        del self._working[path]
        return True

    def files(self) -> dict[str, str]:
        return dict(self._working)

    def commit(self, message: str) -> int:
        """Validate metadata before allocating a snapshot identifier."""
        message = message.strip()
        if not message:
            raise ValueError("commit message is empty")
        commit_id = self._store.save(self._working)
        self._commits[commit_id] = {
            "id": commit_id,
            "message": message,
            "parent": self.head,
        }
        self.head = commit_id
        return commit_id

    def checkout(self, commit_id: int) -> None:
        """Load before changing head so failed lookups are atomic."""
        restored = self._store.load(commit_id)
        self._working = restored
        self.head = commit_id

    def log(self, limit: int | None = None) -> list[dict]:
        """Walk the current branch, which need not include every stored ID."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be nonnegative")
        entries = []
        current = self.head
        while current is not None:
            if limit is not None and len(entries) >= limit:
                break
            record = self._commits[current]
            entries.append(dict(record))
            current = record["parent"]
        return entries
