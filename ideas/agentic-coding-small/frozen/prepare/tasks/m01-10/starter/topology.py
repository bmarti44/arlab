import heapq
from dependency_graph import DependencyGraph

class CycleError(ValueError):
    """Carry both a machine-readable cycle and a stable diagnostic."""

    def __init__(self, cycle):
        raise NotImplementedError()

def find_cycle(graph: DependencyGraph) -> list[str]:
    """Return the first sorted-DFS back edge as a closed cycle path."""
    raise NotImplementedError()

def topological_sort(graph: DependencyGraph) -> list[str]:
    """Use a heap to reconsider lexical priority after each emitted node."""
    raise NotImplementedError()
