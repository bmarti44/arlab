import math
from sparse_vector import SparseVector


def dot(left: SparseVector, right: SparseVector) -> float:
    """Compute an inner product from the intersection of sparse coordinates."""
    if left.dimension != right.dimension:
        raise ValueError("dimension mismatch")
    left_entries = left.to_dict()
    right_entries = right.to_dict()
    if len(left_entries) > len(right_entries):
        left_entries, right_entries = right_entries, left_entries
    total = 0.0
    for index, value in left_entries.items():
        total += value * right_entries.get(index, 0.0)
    return total


def magnitude(vector: SparseVector) -> float:
    """The length of an empty or all-zero vector is zero."""
    squares = 0.0
    for value in vector.to_dict().values():
        squares += value * value
    return math.sqrt(squares)


def cosine_similarity(left: SparseVector, right: SparseVector) -> float:
    """Compare directions, assigning zero similarity to zero vectors."""
    if left.dimension != right.dimension:
        raise ValueError("dimension mismatch")
    left_length = magnitude(left)
    right_length = magnitude(right)
    if left_length == 0 or right_length == 0:
        return 0.0
    product = dot(left, right)
    return product / (left_length * right_length)
