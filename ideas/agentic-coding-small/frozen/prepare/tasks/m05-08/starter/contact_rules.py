def normalize_email(email: str) -> str:
    """Use a deliberately small mailbox rule, not full RFC validation."""
    raise NotImplementedError()

def normalize_phone(phone: str) -> str:
    """Remove presentation punctuation but preserve an international plus."""
    raise NotImplementedError()

def normalize_contact(record: dict) -> dict:
    """Create a canonical contact and retain first occurrences of phones."""
    raise NotImplementedError()
