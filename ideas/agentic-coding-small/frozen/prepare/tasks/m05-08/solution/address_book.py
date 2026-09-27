from contact_rules import normalize_contact, normalize_email


def _copy_contact(contact: dict) -> dict:
    """Copy the one nested mutable field as well as the record itself."""
    return {"name": contact["name"], "email": contact["email"],
            "phones": list(contact["phones"])}


class AddressBook:
    """Merge normalized contacts without sharing caller-owned records."""

    def __init__(self):
        self._contacts = {}

    def _apply(self, incoming: dict) -> dict:
        """Apply an already validated contact to the current collection."""
        email = incoming["email"]
        old = self._contacts.get(email)
        if old is None:
            result = _copy_contact(incoming)
        else:
            result = _copy_contact(old)
            if incoming["name"]:
                result["name"] = incoming["name"]
            for phone in incoming["phones"]:
                if phone not in result["phones"]:
                    result["phones"].append(phone)
        self._contacts[email] = result
        return _copy_contact(result)

    def upsert(self, record: dict) -> dict:
        """Validate a single record before touching stored state."""
        incoming = normalize_contact(record)
        return self._apply(incoming)

    def merge(self, records) -> int:
        """Validate the whole batch, then merge in iterable order."""
        incoming = [normalize_contact(record) for record in records]
        touched = set()
        for contact in incoming:
            self._apply(contact)
            touched.add(contact["email"])
        return len(touched)

    def get(self, email: str) -> dict | None:
        """Normalize a lookup key and return a detached record if present."""
        contact = self._contacts.get(normalize_email(email))
        return None if contact is None else _copy_contact(contact)

    def remove(self, email: str) -> bool:
        key = normalize_email(email)
        if key not in self._contacts:
            return False
        del self._contacts[key]
        return True

    def contacts(self) -> list[dict]:
        """Use email order for a stable listing independent of insert order."""
        return [_copy_contact(self._contacts[email])
                for email in sorted(self._contacts)]
