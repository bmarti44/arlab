def path_counts(rows, cols, blocked):
    raise NotImplementedError


class GridPaths:
    def __init__(self, rows, cols, blocked=()):
        raise NotImplementedError

    def count(self):
        raise NotImplementedError

    def path(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError
