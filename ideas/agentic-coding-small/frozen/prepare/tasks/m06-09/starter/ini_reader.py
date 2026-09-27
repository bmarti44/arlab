def _validate_name(name, section):
    """Validate names shared by the reader and writer."""
    raise NotImplementedError()

def _validate_value(value):
    """Check that a literal string is representable on one INI line."""
    raise NotImplementedError()

def _parse_assignment(line):
    """Only the first equals sign separates the key from the value."""
    raise NotImplementedError()

def loads(text):
    """Read sections and string assignments in insertion order.

    Comments occupy full lines only. Duplicate declarations update
    existing dictionaries, preserving their original position.
    """
    raise NotImplementedError()
