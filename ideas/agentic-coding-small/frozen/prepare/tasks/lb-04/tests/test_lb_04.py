import pytest
from matrix import Matrix


def test_shape_and_copy_boundaries():
    rows = [[1, 2], [3, 4]]
    m = Matrix(rows)
    rows[0][0] = 99
    rows.append([5, 6])
    assert m.shape == (2, 2)
    output = m.to_list()
    output[1][0] = 88
    output.clear()
    assert m.to_list() == [[1, 2], [3, 4]]


def test_invalid_shapes():
    for rows in [[], [[]], [[1], []], [[1, 2], [3]]]:
        with pytest.raises(ValueError):
            Matrix(rows)


def test_add_and_operand_preservation():
    a = Matrix([[1, -2, 0], [3, 4, 5]])
    b = Matrix([[2, 2, 7], [-3, 1, 0]])
    result = a.add(b)
    assert isinstance(result, Matrix)
    assert result.to_list() == [[3, 0, 7], [0, 5, 5]]
    assert a.to_list() == [[1, -2, 0], [3, 4, 5]]
    assert b.to_list() == [[2, 2, 7], [-3, 1, 0]]
    assert result is not a and result is not b


def test_rectangular_multiplication():
    a = Matrix([[1, 2, 3], [-1, 0, 2]])
    b = Matrix([[1, 0], [0, 2], [3, 1]])
    result = a.multiply(b)
    assert result.shape == (2, 2)
    assert result.to_list() == [[10, 7], [5, 2]]
    assert a.to_list() == [[1, 2, 3], [-1, 0, 2]]
    assert b.to_list() == [[1, 0], [0, 2], [3, 1]]


def test_transpose_and_vector_product():
    row = Matrix([[2, -1, 4]])
    column = row.transpose()
    assert column.shape == (3, 1)
    assert column.to_list() == [[2], [-1], [4]]
    assert column.transpose().to_list() == row.to_list()
    assert row.multiply(column).to_list() == [[21]]
    assert Matrix([[7]]).transpose().to_list() == [[7]]


def test_dimension_errors_and_float_arithmetic():
    a = Matrix([[0.5, 1.5]])
    assert a.add(Matrix([[0.5, -0.5]])).to_list() == [[1.0, 1.0]]
    assert a.multiply(Matrix([[2.0], [4.0]])).to_list() == [[7.0]]
    with pytest.raises(ValueError):
        a.add(Matrix([[1], [2]]))
    with pytest.raises(ValueError):
        a.multiply(Matrix([[1, 2]]))
