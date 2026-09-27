import pytest
from dependency_graph import DependencyGraph
from topology import CycleError, find_cycle, topological_sort

def test_empty_and_isolated():
    g = DependencyGraph()
    assert topological_sort(g) == [] and find_cycle(g) == []
    for node in ('z', 'a', 'm', 'a'):
        g.add_node(node)
    assert g.nodes() == ['a', 'm', 'z']
    assert topological_sort(g) == ['a', 'm', 'z']

def test_dependencies_and_dynamic_priority():
    g = DependencyGraph()
    g.add_dependency('b', 'a')
    g.add_dependency('d', 'b')
    g.add_dependency('d', 'c')
    g.add_dependency('b', 'a')
    g.add_node('z')
    assert g.prerequisites('d') == ['b', 'c']
    assert topological_sort(g) == ['a', 'b', 'c', 'd', 'z']
    assert find_cycle(g) == []

def test_cycle_path_and_error():
    g = DependencyGraph()
    for task, prerequisite in [('a', 'b'), ('b', 'c'), ('c', 'b')]:
        g.add_dependency(task, prerequisite)
    assert find_cycle(g) == ['b', 'c', 'b']
    with pytest.raises(CycleError) as error:
        topological_sort(g)
    assert isinstance(error.value, ValueError)
    assert error.value.cycle == ['b', 'c', 'b']
    assert str(error.value) == 'cycle: b -> c -> b'

def test_self_cycle_and_disconnected_component():
    g = DependencyGraph()
    g.add_dependency('b', 'a')
    g.add_dependency('x', 'x')
    assert find_cycle(g) == ['x', 'x']
    with pytest.raises(CycleError) as error:
        topological_sort(g)
    assert error.value.cycle == ['x', 'x']

def test_deterministic_first_cycle():
    g = DependencyGraph()
    for task, prerequisite in [('a', 'd'), ('d', 'd'), ('a', 'b'), ('b', 'c'), ('c', 'b')]:
        g.add_dependency(task, prerequisite)
    assert find_cycle(g) == ['b', 'c', 'b']
    assert find_cycle(g) == ['b', 'c', 'b']

def test_validation_and_detached_queries():
    g = DependencyGraph()
    g.add_dependency('b', 'a')
    g.nodes().clear()
    g.prerequisites('b').clear()
    assert g.nodes() == ['a', 'b'] and g.prerequisites('b') == ['a']
    for action in (lambda: g.add_node(''), lambda: g.add_dependency('new', ''), lambda: g.add_dependency('', 'new')):
        with pytest.raises(ValueError):
            action()
    with pytest.raises(KeyError):
        g.prerequisites('unknown')
    assert g.nodes() == ['a', 'b']
    assert topological_sort(g) == ['a', 'b']
    assert g.prerequisites('b') == ['a']

def test_diamond_and_error_copy():
    g = DependencyGraph()
    for task, prerequisite in [('d', 'b'), ('d', 'c'), ('b', 'a'), ('c', 'a')]:
        g.add_dependency(task, prerequisite)
    assert find_cycle(g) == []
    assert topological_sort(g) == ['a', 'b', 'c', 'd']
    cycle = ['p', 'p']
    error = CycleError(cycle)
    cycle.clear()
    assert error.cycle == ['p', 'p'] and str(error) == 'cycle: p -> p'
