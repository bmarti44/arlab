import pytest
from life import LifeGrid, neighbor_cells


def test_neighbor_geometry():
    assert neighbor_cells(0, 0, 3, 3) == {(1, 0), (0, 1), (1, 1)}
    assert len(neighbor_cells(0, 0, 3, 3, True)) == 8
    assert neighbor_cells(0, 0, 1, 1, True) == set()
    assert neighbor_cells(0, 0, 2, 2, True) == {(1, 0), (0, 1), (1, 1)}


def test_constructor_copies_and_validates():
    cells = [(0, 0), (0, 0)]
    grid = LifeGrid(2, 3, cells)
    cells.append((1, 2))
    assert grid.alive == frozenset({(0, 0)})
    assert isinstance(grid.alive, frozenset)
    assert (grid.width, grid.height, grid.wrap) == (2, 3, False)
    for args in [(0, 2), (2, -1), (2, 2, [(2, 0)]), (2, 2, [(-1, 0)])]:
        with pytest.raises(ValueError):
            LifeGrid(*args)


def test_blinker_and_simultaneous_update():
    grid = LifeGrid(5, 5, [(1, 2), (2, 2), (3, 2)])
    next_grid = grid.step()
    assert next_grid.alive == frozenset({(2, 1), (2, 2), (2, 3)})
    assert grid.alive == frozenset({(1, 2), (2, 2), (3, 2)})
    assert next_grid is not grid
    assert grid.advance(2).alive == grid.alive


def test_still_life_and_isolation():
    square = {(1, 1), (1, 2), (2, 1), (2, 2)}
    assert LifeGrid(4, 4, square).advance(4).alive == square
    assert LifeGrid(3, 3, [(1, 1)]).step().alive == frozenset()
    assert LifeGrid(3, 3).step().alive == frozenset()


def test_wraparound_changes_births():
    cells = {(4, 2), (0, 2), (1, 2)}
    wrapped = LifeGrid(5, 5, cells, True).step()
    assert wrapped.alive == {(0, 1), (0, 2), (0, 3)}
    assert (wrapped.width, wrapped.height, wrapped.wrap) == (5, 5, True)
    assert LifeGrid(5, 5, cells).step().alive == frozenset()


def test_tiny_torus_counts_distinct_cells():
    assert LifeGrid(1, 1, [(0, 0)], True).step().alive == frozenset()
    assert LifeGrid(1, 2, [(0, 0), (0, 1)], True).step().alive == frozenset()
    tiny = LifeGrid(2, 2, [(0, 0), (1, 0), (0, 1)], True)
    assert tiny.step().alive == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_advance_zero_and_negative():
    grid = LifeGrid(3, 4, [(1, 1)], True)
    copy = grid.advance(0)
    assert copy is not grid and copy.alive == grid.alive
    assert (copy.width, copy.height, copy.wrap) == (3, 4, True)
    with pytest.raises(ValueError):
        grid.advance(-1)
