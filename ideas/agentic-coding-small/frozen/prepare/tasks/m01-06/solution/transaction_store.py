def diff(before, after):
    """Describe changes without treating a stored None as a missing key."""
    added = {}
    updated = {}
    removed = {}
    for key, value in after.items():
        if key not in before:
            added[key] = value
        elif before[key] != value:
            updated[key] = (before[key], value)
    for key, value in before.items():
        if key not in after:
            removed[key] = value
    return {"added": added, "updated": updated, "removed": removed}


class TransactionStore:
    """Each active transaction owns a shallow copy of its parent's view.

    Replacing a parent on commit preserves the next outer rollback boundary.
    Keeping the base view as frame zero makes writes use one consistent path.
    """

    def __init__(self, initial=None):
        if initial is None:
            initial = {}
        base = dict(initial)
        self._frames = [base]

    def set(self, key, value):
        """Write to the innermost visible view."""
        current = self._frames[-1]
        current[key] = value

    def get(self, key):
        """Dictionary lookup naturally distinguishes missing keys and None."""
        current = self._frames[-1]
        return current[key]

    def delete(self, key):
        """Pop validates existence before changing the current mapping."""
        current = self._frames[-1]
        return current.pop(key)

    def snapshot(self):
        """Detach the mapping while preserving the identity of its values."""
        return dict(self._frames[-1])

    def begin(self):
        """Push a new rollback boundary, even if no changes are made."""
        current = self._frames[-1]
        self._frames.append(dict(current))
        depth = len(self._frames) - 1
        return depth

    def commit(self):
        """Merge the child view into its parent by replacing the parent frame."""
        if len(self._frames) == 1:
            raise RuntimeError("no transaction")
        child = self._frames.pop()
        self._frames[-1] = child
        depth = len(self._frames) - 1
        return depth

    def rollback(self):
        """Discard exactly one active transaction."""
        if len(self._frames) == 1:
            raise RuntimeError("no transaction")
        self._frames.pop()
        depth = len(self._frames) - 1
        return depth
