class DependencyGraph:
    """Store each task's prerequisites, including explicitly isolated tasks."""

    def __init__(self):
        self._dependencies = {}

    def add_node(self, name):
        """Adding an existing name leaves its prerequisites intact."""
        if name == "":
            raise ValueError("empty node name")
        self._dependencies.setdefault(name, set())

    def add_dependency(self, task, prerequisite):
        """Validate both names before making either node visible."""
        if task == "" or prerequisite == "":
            raise ValueError("empty node name")
        self.add_node(task)
        self.add_node(prerequisite)
        self._dependencies[task].add(prerequisite)

    def nodes(self):
        """Return a detached, deterministic list of all known nodes."""
        return sorted(self._dependencies)

    def prerequisites(self, node):
        """Keep internal sets private so queries cannot alter graph edges."""
        return sorted(self._dependencies[node])
