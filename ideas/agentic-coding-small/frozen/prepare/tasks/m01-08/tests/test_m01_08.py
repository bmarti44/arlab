import pytest
from sparse_vector import SparseVector
from vector_math import dot, magnitude, cosine_similarity

def test_storage_and_detached_copies():
    entries = {1: 3, 4: 0}
    v = SparseVector(5, entries)
    entries[1] = 99
    assert v.dimension == 5 and v.to_dict() == {1: 3.0}
    assert type(v[1]) is float and v[0] == 0.0
    exported = v.to_dict()
    exported[2] = 8
    assert v[2] == 0.0

def test_bounds_and_zero_dimension():
    v = SparseVector(0)
    assert v.to_dict() == {} and magnitude(v) == 0.0
    with pytest.raises(ValueError):
        SparseVector(-1)
    for index in (-1, 3):
        with pytest.raises(IndexError):
            SparseVector(3, {index: 0})
        with pytest.raises(IndexError):
            SparseVector(3)[index]
    with pytest.raises(IndexError):
        v[0]

def test_add_cancellation_and_independence():
    a = SparseVector(4, {0: 2, 2: 5})
    b = SparseVector(4, {0: -2, 1: 3})
    c = a.add(b)
    assert c.to_dict() == {1: 3.0, 2: 5.0} and c.dimension == 4
    assert a.to_dict() == {0: 2.0, 2: 5.0}
    assert b.to_dict() == {0: -2.0, 1: 3.0}
    assert c is not a and a.add(a).to_dict() == {0: 4.0, 2: 10.0}

def test_scaling_zero_and_negative():
    a = SparseVector(4, {0: 2, 3: -3})
    assert a.scale(-0.5).to_dict() == {0: -1.0, 3: 1.5}
    zero = a.scale(0)
    assert zero.dimension == 4 and zero.to_dict() == {}
    assert a.scale(1) is not a and a.to_dict() == {0: 2.0, 3: -3.0}

def test_dot_and_magnitude():
    a = SparseVector(5, {0: 3, 4: 4})
    b = SparseVector(5, {0: -2, 1: 9})
    assert dot(a, b) == -6.0 and dot(b, a) == -6.0
    assert magnitude(a) == 5.0
    assert cosine_similarity(a, b) == pytest.approx(-6 / (5 * 85 ** 0.5))

def test_cosine_zero_opposite_and_mismatch():
    a = SparseVector(2, {0: 2})
    assert cosine_similarity(a, a) == pytest.approx(1)
    assert cosine_similarity(a, a.scale(-1)) == pytest.approx(-1)
    assert cosine_similarity(a, SparseVector(2)) == 0.0
    assert cosine_similarity(SparseVector(0), SparseVector(0)) == 0.0
    for action in (lambda: a.add(SparseVector(3)), lambda: dot(a, SparseVector(0)), lambda: cosine_similarity(SparseVector(2), SparseVector(3))):
        with pytest.raises(ValueError):
            action()

def test_large_sparse_dimension():
    a = SparseVector(10**9, {2: 3, 999999999: 4})
    b = SparseVector(10**9, {2: 2, 8: 6})
    assert dot(a, b) == 6
    assert magnitude(a) == 5
    assert a.add(b).to_dict() == {2: 5.0, 8: 6.0, 999999999: 4.0}
