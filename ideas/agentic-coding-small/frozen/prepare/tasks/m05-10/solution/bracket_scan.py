PAIRS = {"(": ")", "[": "]", "{": "}"}


def scan_line(text: str, line_number: int, stack: list):
    """Track brackets outside comments and single-line quoted strings."""
    quote = None
    escaped = False
    for column, character in enumerate(text, start=1):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character == "#":
            break
        if character in ("'", '"'):
            quote = character
            continue
        if character in PAIRS:
            stack.append((character, line_number, column))
            continue
        if character not in PAIRS.values():
            continue
        if not stack:
            return line_number, column, f"unexpected closing {character}"
        expected = PAIRS[stack[-1][0]]
        if character != expected:
            return line_number, column, f"expected {expected}, got {character}"
        stack.pop()
    if quote is not None:
        return line_number, len(text) + 1, "unterminated string"
    return None


def finish(stack: list):
    """Report the innermost unresolved opener without consuming it."""
    if not stack:
        return None
    character, line, column = stack[-1]
    return line, column, f"unclosed {character}"
