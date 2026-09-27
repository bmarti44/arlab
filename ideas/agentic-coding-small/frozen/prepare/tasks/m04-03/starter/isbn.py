def _compact(text):
    """Discard the two permitted separator characters."""
    if not isinstance(text, str):
        raise TypeError('ISBN must be text')
    return text.replace(' ', '').replace('-', '').upper()


def _ascii_digits(text):
    return all('0' <= character <= '9' for character in text)


def check_digit10(prefix):
    """Return the ISBN-10 check character for nine digits."""
    digits = _compact(prefix)
    if len(digits) != 9 or not _ascii_digits(digits):
        raise ValueError('expected nine ASCII digits')
    total = sum((9 - index) * int(char) for index, char in enumerate(digits))
    value = (-total) % 11
    return 'X' if value == 10 else str(value)


def check_digit13(prefix):
    """Return the ISBN-13 check character for twelve digits."""
    digits = _compact(prefix)
    if len(digits) != 12 or not _ascii_digits(digits):
        raise ValueError('expected twelve ASCII digits')
    total = sum(int(char) * (3 if index % 2 == 0 else 1)
                for index, char in enumerate(digits))
    return str((-total) % 10)


def is_valid(text):
    """Validate shape and checksum, without registration-prefix rules."""
    value = _compact(text)
    if len(value) == 10:
        if not _ascii_digits(value[:9]):
            return False
        if value[-1] not in '0123456789X':
            return False
        return value[-1] == check_digit10(value[:9])
    if len(value) == 13:
        if not _ascii_digits(value):
            return False
        return value[-1] == check_digit13(value[:12])
    return False


def normalize(text):
    """Return compact uppercase text only for a valid ISBN."""
    value = _compact(text)
    if not is_valid(value):
        raise ValueError('invalid ISBN')
    return value


def to_isbn13(text):
    """Convert ISBN-10 through the 978 namespace; preserve ISBN-13."""
    value = normalize(text)
    if len(value) == 13:
        return value
    prefix = '978' + value[:9]
    return prefix + check_digit13(prefix)
