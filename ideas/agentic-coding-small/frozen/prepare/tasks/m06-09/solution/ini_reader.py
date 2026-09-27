def _validate_name(name, section):
    """Validate names shared by the reader and writer."""
    if not name or name != name.strip():
        raise ValueError('name must be nonempty and stripped')
    if '\r' in name or '\n' in name:
        raise ValueError('multiline names are unsupported')
    if section:
        if '[' in name or ']' in name:
            raise ValueError('bracket in section name')
    elif '=' in name or name[0] in '[#;':
        raise ValueError('invalid key')


def _validate_value(value):
    """Check that a literal string is representable on one INI line."""
    if value != value.strip():
        raise ValueError('value must already be stripped')
    if '\r' in value or '\n' in value:
        raise ValueError('multiline values are unsupported')


def _parse_assignment(line):
    """Only the first equals sign separates the key from the value."""
    if '=' not in line:
        raise ValueError('assignment requires equals sign')
    key, value = line.split('=', 1)
    key = key.strip()
    value = value.strip()
    _validate_name(key, False)
    _validate_value(value)
    return key, value


def loads(text):
    """Read sections and string assignments in insertion order.

    Comments occupy full lines only. Duplicate declarations update
    existing dictionaries, preserving their original position.
    """
    result = {}
    current = None
    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if not line or line[0] in '#;':
            continue
        if line.startswith('['):
            if not line.endswith(']'):
                raise ValueError('invalid section header')
            name = line[1:-1].strip()
            _validate_name(name, True)
            current = result.setdefault(name, {})
            continue
        if current is None:
            raise ValueError('key before first section')
        key, value = _parse_assignment(line)
        current[key] = value
    return result
