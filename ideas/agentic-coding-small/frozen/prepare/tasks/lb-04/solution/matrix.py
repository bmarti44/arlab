class Matrix:
    """A rectangular, nonempty matrix with independent row storage."""

    def __init__(self, rows):
        if not rows or not rows[0]:
            raise ValueError("matrix must be nonempty")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("rows must have equal length")
        self._rows = [list(row) for row in rows]

    @property
    def shape(self):
        """Return (row count, column count)."""
        return len(self._rows), len(self._rows[0])

    def to_list(self):
        """Return a copy of both the outer list and each row."""
        return [row[:] for row in self._rows]

    def add(self, other):
        """Add matching matrices without changing either operand."""
        if self.shape != other.shape:
            raise ValueError("incompatible shapes for addition")
        height, width = self.shape
        result = []
        for row in range(height):
            result.append([
                self._rows[row][col] + other._rows[row][col]
                for col in range(width)
            ])
        return Matrix(result)

    def multiply(self, other):
        """Perform matrix multiplication, not elementwise multiplication."""
        height, inner = self.shape
        other_height, width = other.shape
        if inner != other_height:
            raise ValueError("incompatible shapes for multiplication")
        result = []
        for row in range(height):
            output_row = []
            for col in range(width):
                total = sum(self._rows[row][k] * other._rows[k][col]
                            for k in range(inner))
                output_row.append(total)
            result.append(output_row)
        return Matrix(result)

    def transpose(self):
        """Return a new matrix with rows and columns exchanged."""
        height, width = self.shape
        rows = [[self._rows[row][col] for row in range(height)]
                for col in range(width)]
        return Matrix(rows)
