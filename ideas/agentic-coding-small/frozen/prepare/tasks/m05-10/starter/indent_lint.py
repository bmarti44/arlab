from bracket_scan import scan_line, finish

def lint(text: str, indent_width: int=4):
    """Return the earliest line error, then any outstanding bracket error.

    Continuation lines retain the surrounding block's indentation level.
    Comment-only and blank lines have no effect on either stack.
    """
    raise NotImplementedError()
