import math


class Vector2:
    """A two-dimensional vector that also serves as a point.

    Methods return new vectors, making instances safe to reuse in
    a sequence of geometric calculations.
    """

    def __init__(self, x=0.0, y=0.0):
        """Store both coordinates as floats."""
        self.x = float(x)
        self.y = float(y)

    def add(self, other):
        """Return the componentwise sum without changing either input."""
        x = self.x + other.x
        y = self.y + other.y
        return Vector2(x, y)

    def scale(self, factor):
        """Return a new vector, even for a factor of one or zero."""
        x = self.x * factor
        y = self.y * factor
        return Vector2(x, y)

    def dot(self, other):
        """Return the scalar dot product."""
        horizontal = self.x * other.x
        vertical = self.y * other.y
        return horizontal + vertical

    def distance_to(self, other):
        """Interpret the vectors as points and measure their distance."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.hypot(dx, dy)

    def normalized(self):
        """Return a fresh unit vector, rejecting the zero vector."""
        length = math.hypot(self.x, self.y)
        if length == 0.0:
            raise ValueError('the zero vector has no direction')
        return Vector2(self.x / length, self.y / length)
