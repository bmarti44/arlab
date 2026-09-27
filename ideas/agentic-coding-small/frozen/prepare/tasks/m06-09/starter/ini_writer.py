from ini_reader import _validate_name, _validate_value

def dumps(data):
    """Serialize a representable nested dictionary deterministically.

    Values are literal strings. Refusing surrounding whitespace and
    line breaks ensures the reader can reconstruct every value.
    Empty sections are retained because their presence is meaningful.
    """
    raise NotImplementedError()
