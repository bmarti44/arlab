import pytest
from bracket_scan import scan_line, finish
from indent_lint import lint


def test_scanner_stack_across_lines_and_finish():
    stack = []
    assert scan_line("x = ([", 3, stack) is None
    assert stack == [("(", 3, 5), ("[", 3, 6)]
    assert finish(stack) == (3, 6, "unclosed [")
    assert len(stack) == 2
    assert scan_line("])", 4, stack) is None
    assert stack == [] and finish(stack) is None


def test_closing_errors_and_locations():
    assert lint("x = )") == (1, 5, "unexpected closing )")
    assert lint("x = ([)]") == (1, 7, "expected ], got )")
    stack = [("{", 2, 4)]
    assert scan_line("]", 5, stack) == (5, 1, "expected }, got ]")
    assert stack == [("{", 2, 4)]


def test_quotes_escapes_and_comments():
    assert lint('x = "[ # ]" # (' + "\n" + "y = (']')") is None
    escaped = 'x = "a' + chr(92) + '"["'
    assert lint(escaped) is None
    assert lint("x = 'oops") == (1, 10, "unterminated string")
    assert lint(") 'oops") == (1, 1, "unexpected closing )")
    assert lint('x = "a' + chr(92)) == (1, 8, "unterminated string")


def test_indentation_stack_and_skipped_levels():
    assert lint("root\n        deep\n            deeper\n        deep\nroot") is None
    assert lint("root\n        deep\n    unseen") == (3, 1, "inconsistent dedent")
    assert lint("    first\n        second") is None
    assert lint("x\n  y\nx", 2) is None


def test_indent_errors_precede_bracket_scan():
    assert lint("ok\n  ]") == (2, 1, "indentation must be a multiple of 4")
    assert lint("ok\n  \t]") == (2, 3, "tab indentation")
    assert lint("x\n   y", 2) == (2, 1, "indentation must be a multiple of 2")


def test_bracket_continuations_skip_indentation():
    assert lint("root\n    call(\n \t [1, 2],\n )\n    next\nend") is None
    assert lint("call(\n  ]") == (2, 3, "expected ), got ]")
    assert lint("root\n    call(\n )\n  next") == (4, 1, "indentation must be a multiple of 4")


def test_blank_comment_lines_and_eof():
    assert lint("") is None
    assert lint(" \t\n  \t# ] '\nroot\n    leaf\n") is None
    assert lint("call(\n # ]\n [") == (3, 2, "unclosed [")
    assert lint("(\n'bad") == (2, 5, "unterminated string")


def test_invalid_widths():
    for width in [0, -1, True, 2.0, "4"]:
        with pytest.raises(ValueError):
            lint("", width)
