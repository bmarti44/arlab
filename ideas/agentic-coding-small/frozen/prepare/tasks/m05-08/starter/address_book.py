from contact_rules import normalize_contact, normalize_email

def _copy_contact(contact: dict) -> dict:
    """Copy the one nested mutable field as well as the record itself."""
    raise NotImplementedError()

class AddressBook:
    """Merge normalized contacts without sharing caller-owned records."""

    def __init__(self):
        raise NotImplementedError()

    def _apply(self, incoming: dict) -> dict:
        """Apply an already validated contact to the current collection."""
        raise NotImplementedError()

    def upsert(self, record: dict) -> dict:
        """Validate a single record before touching stored state."""
        raise NotImplementedError()

    def merge(self, records) -> int:
        """Validate the whole batch, then merge in iterable order."""
        raise NotImplementedError()

    def get(self, email: str) -> dict | None:
        """Normalize a lookup key and return a detached record if present."""
        raise NotImplementedError()

    def remove(self, email: str) -> bool:
        raise NotImplementedError()

    def contacts(self) -> list[dict]:
        """Use email order for a stable listing independent of insert order."""
        raise NotImplementedError()
