def normalize_email(email: str) -> str:
    """Use a deliberately small mailbox rule, not full RFC validation."""
    result = email.strip().lower()
    if result.count("@") != 1:
        raise ValueError("email needs one at-sign")
    local, domain = result.split("@")
    if not local or not domain or any(char.isspace() for char in result):
        raise ValueError("invalid email")
    return result


def normalize_phone(phone: str) -> str:
    """Remove presentation punctuation but preserve an international plus."""
    result = "".join(char for char in phone if char not in " ()-.")
    digits = result[1:] if result.startswith("+") else result
    if not 1 <= len(digits) <= 15:
        raise ValueError("phone needs 1 to 15 digits")
    if any(char not in "0123456789" for char in digits):
        raise ValueError("invalid phone character")
    return result


def normalize_contact(record: dict) -> dict:
    """Create a canonical contact and retain first occurrences of phones."""
    if "email" not in record:
        raise ValueError("email is required")
    email = normalize_email(record["email"])
    name = record.get("name", "").strip()
    phones = []
    seen = set()
    for phone in record.get("phones", []):
        normalized = normalize_phone(phone)
        if normalized not in seen:
            seen.add(normalized)
            phones.append(normalized)
    return {"name": name, "email": email, "phones": phones}
