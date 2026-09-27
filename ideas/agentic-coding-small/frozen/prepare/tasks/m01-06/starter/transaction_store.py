def diff(before, after):
    """Describe changes without treating a stored None as a missing key."""
    raise NotImplementedError()

class TransactionStore:
    """Each active transaction owns a shallow copy of its parent's view.

    Replacing a parent on commit preserves the next outer rollback boundary.
    Keeping the base view as frame zero makes writes use one consistent path.
    """

    def __init__(self, initial=None):
        raise NotImplementedError()

    def set(self, key, value):
        """Write to the innermost visible view."""
        raise NotImplementedError()

    def get(self, key):
        """Dictionary lookup naturally distinguishes missing keys and None."""
        raise NotImplementedError()

    def delete(self, key):
        """Pop validates existence before changing the current mapping."""
        raise NotImplementedError()

    def snapshot(self):
        """Detach the mapping while preserving the identity of its values."""
        raise NotImplementedError()

    def begin(self):
        """Push a new rollback boundary, even if no changes are made."""
        raise NotImplementedError()

    def commit(self):
        """Merge the child view into its parent by replacing the parent frame."""
        raise NotImplementedError()

    def rollback(self):
        """Discard exactly one active transaction."""
        raise NotImplementedError()
