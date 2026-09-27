def weighted_mean(pairs):
    """Average a stream of values, rejecting nonpositive weights."""
    raise NotImplementedError()

class Gradebook:
    """Keep assignment percentages separate from category weights."""

    def __init__(self, weights):
        raise NotImplementedError()

    def add(self, category, score, possible=100):
        """Validate completely before appending an assignment."""
        raise NotImplementedError()

    def category_average(self, category):
        """Every assignment has equal weight within its category."""
        raise NotImplementedError()

    def overall(self):
        """Only categories with assignments participate in the average."""
        raise NotImplementedError()

    def drop_lowest(self, category, count=1):
        """Select by percentage while retaining insertion order for survivors."""
        raise NotImplementedError()

    def report(self):
        """Return a detached report, retaining the configured category order."""
        raise NotImplementedError()
