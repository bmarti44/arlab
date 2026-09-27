import heapq
from dependency_graph import DependencyGraph


class CycleError(ValueError):
    """Carry both a machine-readable cycle and a stable diagnostic."""

    def __init__(self, cycle):
        self.cycle = list(cycle)
        super().__init__("cycle: " + " -> ".join(self.cycle))


def find_cycle(graph: DependencyGraph) -> list[str]:
    """Return the first sorted-DFS back edge as a closed cycle path."""
    completed = set()
    active = {}
    path = []

    def visit(node):
        if node in active:
            return path[active[node]:] + [node]
        if node in completed:
            return []
        active[node] = len(path)
        path.append(node)
        for prerequisite in graph.prerequisites(node):
            cycle = visit(prerequisite)
            if cycle:
                return cycle
        path.pop()
        del active[node]
        completed.add(node)
        return []

    for node in graph.nodes():
        cycle = visit(node)
        if cycle:
            return cycle
    return []


def topological_sort(graph: DependencyGraph) -> list[str]:
    """Use a heap to reconsider lexical priority after each emitted node."""
    nodes = graph.nodes()
    remaining = {}
    dependents = {node: [] for node in nodes}
    for node in nodes:
        prerequisites = graph.prerequisites(node)
        remaining[node] = len(prerequisites)
        for prerequisite in prerequisites:
            dependents[prerequisite].append(node)
    available = [node for node in nodes if remaining[node] == 0]
    heapq.heapify(available)
    ordered = []
    while available:
        node = heapq.heappop(available)
        ordered.append(node)
        for dependent in dependents[node]:
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                heapq.heappush(available, dependent)
    if len(ordered) != len(nodes):
        raise CycleError(find_cycle(graph))
    return ordered
