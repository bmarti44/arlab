from bracket_scan import scan_line, finish


def lint(text: str, indent_width: int = 4):
    """Return the earliest line error, then any outstanding bracket error.

    Continuation lines retain the surrounding block's indentation level.
    Comment-only and blank lines have no effect on either stack.
    """
    if type(indent_width) is not int or indent_width <= 0:
        raise ValueError("indent width must be a positive integer")
    brackets = []
    indents = [0]
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if not brackets:
            prefix_length = 0
            for character in line:
                if character not in " \t":
                    break
                prefix_length += 1
            prefix = line[:prefix_length]
            if "\t" in prefix:
                return line_number, prefix.index("\t") + 1, "tab indentation"
            level = len(prefix)
            if level % indent_width:
                message = f"indentation must be a multiple of {indent_width}"
                return line_number, 1, message
            if level > indents[-1]:
                indents.append(level)
            elif level < indents[-1]:
                if level not in indents:
                    return line_number, 1, "inconsistent dedent"
                while indents[-1] > level:
                    indents.pop()
        error = scan_line(line, line_number, brackets)
        if error is not None:
            return error
    return finish(brackets)
