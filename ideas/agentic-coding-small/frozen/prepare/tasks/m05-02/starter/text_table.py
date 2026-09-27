class TextTable:
    """Store text cells and calculate widths when rendering."""

    def __init__(self, headers, alignments=None):
        raise NotImplementedError()

    def add_row(self, values) -> None:
        """Validate the entire row before modifying the table."""
        raise NotImplementedError()

    def row_count(self) -> int:
        """Return the number of stored data rows."""
        raise NotImplementedError()

    def clear(self) -> None:
        """Discard data so future renders recalculate narrower widths."""
        raise NotImplementedError()

    def render(self) -> str:
        """Render headers, a rule, and all rows without a final newline."""
        raise NotImplementedError()
