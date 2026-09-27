from ini_reader import _validate_name, _validate_value


def dumps(data):
    """Serialize a representable nested dictionary deterministically.

    Values are literal strings. Refusing surrounding whitespace and
    line breaks ensures the reader can reconstruct every value.
    Empty sections are retained because their presence is meaningful.
    """
    sections = []
    for name, entries in data.items():
        _validate_name(name, True)
        lines = ['[' + name + ']']
        for key, value in entries.items():
            _validate_name(key, False)
            _validate_value(value)
            lines.append(key + '=' + value)
        sections.append('\n'.join(lines))

    if not sections:
        return ''
    # Joining complete sections gives one blank line between them.
    return '\n\n'.join(sections) + '\n'
