import math

class Vector2:
    """A two-dimensional vector that also serves as a point.

    Methods return new vectors, making instances safe to reuse in
    a sequence of geometric calculations.
    """

    def __init__(self, x=0.0, y=0.0):
        """Store both coordinates as floats."""
        raise NotImplementedError()

    def add(self, other):
        """Return the componentwise sum without changing either input."""
        raise NotImplementedError()

    def scale(self, factor):
        """Return a new vector, even for a factor of one or zero."""
        raise NotImplementedError()

    def dot(self, other):
        """Return the scalar dot product."""
        raise NotImplementedError()

    def distance_to(self, other):
        """Interpret the vectors as points and measure their distance."""
        raise NotImplementedError()

    def normalized(self):
        """Return a fresh unit vector, rejecting the zero vector."""
        raise NotImplementedError()
