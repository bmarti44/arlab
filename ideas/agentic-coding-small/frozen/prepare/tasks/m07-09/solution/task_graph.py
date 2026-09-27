"""Validation and normalization of dependency records."""


def build_graph(records: list[dict]) -> tuple[dict[str, int], dict[str, set[str]]]:
    durations = {}
    dependencies = {}
    for record in records:
        if any(field not in record for field in ("id", "duration", "deps")):
            raise ValueError("missing task field")
        name = record["id"]
        duration = record["duration"]
        deps = record["deps"]
        if not isinstance(name, str) or not name:
            raise ValueError("invalid task id")
        if name in durations:
            raise ValueError("duplicate task id")
        if not isinstance(duration, int) or isinstance(duration, bool) or duration < 0:
            raise ValueError("invalid duration")
        if not isinstance(deps, list) or any(not isinstance(dep, str) for dep in deps):
            raise ValueError("invalid dependency list")
        durations[name] = duration
        dependencies[name] = set(deps)

    for name, deps in dependencies.items():
        if name in deps:
            raise ValueError("self dependency")
        if any(dep not in durations for dep in deps):
            raise ValueError("unknown dependency")
    return durations, dependencies
