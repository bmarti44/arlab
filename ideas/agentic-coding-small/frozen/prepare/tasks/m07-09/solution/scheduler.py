"""Earliest-start scheduling with deterministic topological ordering."""
import heapq
from task_graph import build_graph


def schedule(records: list[dict]) -> dict:
    durations, dependencies = build_graph(records)
    remaining = {name: len(deps) for name, deps in dependencies.items()}
    followers = {name: [] for name in durations}
    for name, deps in dependencies.items():
        for dep in deps:
            followers[dep].append(name)
    ready = [name for name, count in remaining.items() if count == 0]
    heapq.heapify(ready)
    order = []
    times = {}
    predecessor = {}

    while ready:
        name = heapq.heappop(ready)
        deps = dependencies[name]
        if deps:
            parent = min(deps, key=lambda dep: (-times[dep][1], dep))
            start = times[parent][1]
        else:
            parent = None
            start = 0
        predecessor[name] = parent
        times[name] = (start, start + durations[name])
        order.append(name)
        for child in followers[name]:
            remaining[child] -= 1
            if remaining[child] == 0:
                heapq.heappush(ready, child)

    if len(order) != len(durations):
        raise ValueError("dependency cycle")
    if not order:
        return {
            "order": [],
            "times": {},
            "makespan": 0,
            "critical_path": [],
        }

    last = min(times, key=lambda name: (-times[name][1], name))
    makespan = times[last][1]
    critical_path = []
    current = last
    while current is not None:
        critical_path.append(current)
        current = predecessor[current]
    critical_path.reverse()
    return {
        "order": order,
        "times": times,
        "makespan": makespan,
        "critical_path": critical_path,
    }
