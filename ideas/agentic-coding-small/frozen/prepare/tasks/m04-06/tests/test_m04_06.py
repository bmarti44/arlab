from math import comb
import pytest
from grid_paths import GridPaths, path_counts


def test_counts_and_right_preference():
    grid = GridPaths(3, 3)
    assert grid.count() == 6
    assert grid.path() == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    assert grid.render() == 'S**\n..*\n..E'
    assert path_counts(2, 3, []) == [[3, 2, 1], [1, 1, 1]]


def test_obstacle_detour_and_dead_end():
    grid = GridPaths(3, 3, [(0, 2), (1, 1)])
    assert grid.count() == 1
    assert grid.path() == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert grid.render() == 'S.#\n*#.\n**E'
    assert path_counts(3, 3, [(0, 2), (1, 1)])[0][1] == 0


def test_unreachable_and_blocked_endpoints():
    grid = GridPaths(2, 2, [(0, 1), (1, 0)])
    assert grid.count() == 0 and grid.path() is None
    assert grid.render() == 'S#\n#E'
    assert GridPaths(2, 2, [(0, 0)]).render() == '#.\n.E'
    blocked_end = GridPaths(2, 2, [(1, 1)])
    assert blocked_end.count() == 0
    assert blocked_end.render() == 'S.\n.#'


def test_singleton_and_thin_grids():
    grid = GridPaths(1, 1)
    assert grid.count() == 1 and grid.path() == [(0, 0)]
    assert grid.render() == 'S'
    blocked = GridPaths(1, 1, [(0, 0)])
    assert blocked.count() == 0 and blocked.path() is None
    assert blocked.render() == '#'
    assert GridPaths(1, 4).render() == 'S**E'
    assert GridPaths(4, 1).render() == 'S\n*\n*\nE'


def test_copy_duplicates_and_fresh_path():
    walls = [(1, 1), (1, 1)]
    grid = GridPaths(3, 3, iter(walls))
    walls.append((0, 0))
    assert grid.count() == 2
    path = grid.path()
    path.clear()
    assert len(grid.path()) == 5
    assert path_counts(2, 2, iter([(0, 1), (0, 1)])) == [[1, 0], [1, 1]]


def test_validation():
    for rows, cols, walls in [(0, 2, []), (2, -1, []), (2, 2, [(-1, 0)]), (2, 2, [(0, 2)])]:
        with pytest.raises(ValueError):
            GridPaths(rows, cols, walls)
        with pytest.raises(ValueError):
            path_counts(rows, cols, walls)


def test_large_count():
    grid = GridPaths(30, 30)
    assert grid.count() == comb(58, 29)
    assert len(grid.path()) == 59
