class DependencyGraph:
    """Store each task's prerequisites, including explicitly isolated tasks."""

    def __init__(self):
        raise NotImplementedError()

    def add_node(self, name):
        """Adding an existing name leaves its prerequisites intact."""
        raise NotImplementedError()

    def add_dependency(self, task, prerequisite):
        """Validate both names before making either node visible."""
        raise NotImplementedError()

    def nodes(self):
        """Return a detached, deterministic list of all known nodes."""
        raise NotImplementedError()

    def prerequisites(self, node):
        """Keep internal sets private so queries cannot alter graph edges."""
        raise NotImplementedError()
