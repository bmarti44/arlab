import math
from sparse_vector import SparseVector

def dot(left: SparseVector, right: SparseVector) -> float:
    """Compute an inner product from the intersection of sparse coordinates."""
    raise NotImplementedError()

def magnitude(vector: SparseVector) -> float:
    """The length of an empty or all-zero vector is zero."""
    raise NotImplementedError()

def cosine_similarity(left: SparseVector, right: SparseVector) -> float:
    """Compare directions, assigning zero similarity to zero vectors."""
    raise NotImplementedError()
