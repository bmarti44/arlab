class TextTable:
    """Store text cells and calculate widths when rendering."""

    def __init__(self, headers, alignments=None):
        self.headers = [str(value) for value in headers]
        if not self.headers:
            raise ValueError("at least one column is required")
        if any("\n" in cell or "\r" in cell for cell in self.headers):
            raise ValueError("headers must be single-line")
        if alignments is None:
            alignments = ["left"] * len(self.headers)
        if len(alignments) != len(self.headers):
            raise ValueError("alignment count must match columns")
        if any(alignment not in ("left", "right") for alignment in alignments):
            raise ValueError("unknown alignment")
        self.alignments = list(alignments)
        self.rows = []

    def add_row(self, values) -> None:
        """Validate the entire row before modifying the table."""
        if len(values) != len(self.headers):
            raise ValueError("row has the wrong number of cells")
        cells = [str(value) for value in values]
        if any("\n" in cell or "\r" in cell for cell in cells):
            raise ValueError("cells must be single-line")
        self.rows.append(cells)

    def row_count(self) -> int:
        """Return the number of stored data rows."""
        return len(self.rows)

    def clear(self) -> None:
        """Discard data so future renders recalculate narrower widths."""
        self.rows.clear()

    def render(self) -> str:
        """Render headers, a rule, and all rows without a final newline."""
        rows = [self.headers] + self.rows
        widths = [max(len(row[index]) for row in rows)
                  for index in range(len(self.headers))]
        lines = []
        for row_index, row in enumerate(rows):
            padded = []
            for cell, width, alignment in zip(row, widths, self.alignments):
                if alignment == "right":
                    padded.append(cell.rjust(width))
                else:
                    padded.append(cell.ljust(width))
            lines.append(" | ".join(padded))
            if row_index == 0:
                lines.append("-+-".join("-" * width for width in widths))
        return "\n".join(lines)
