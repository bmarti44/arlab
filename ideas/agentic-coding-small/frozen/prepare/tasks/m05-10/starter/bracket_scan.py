PAIRS = {'(': ')', '[': ']', '{': '}'}

def scan_line(text: str, line_number: int, stack: list):
    """Track brackets outside comments and single-line quoted strings."""
    raise NotImplementedError()

def finish(stack: list):
    """Report the innermost unresolved opener without consuming it."""
    raise NotImplementedError()
