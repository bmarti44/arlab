def weighted_mean(pairs):
    """Average a stream of values, rejecting nonpositive weights."""
    total = 0.0
    total_weight = 0.0
    for value, weight in pairs:
        if weight <= 0:
            raise ValueError("weight must be positive")
        total += value * weight
        total_weight += weight
    if total_weight == 0:
        return None
    return total / total_weight


class Gradebook:
    """Keep assignment percentages separate from category weights."""

    def __init__(self, weights):
        if not weights:
            raise ValueError("at least one category is required")
        if any(weight <= 0 for weight in weights.values()):
            raise ValueError("weights must be positive")
        self._weights = dict(weights)
        self._records = {name: [] for name in weights}

    def add(self, category, score, possible=100):
        """Validate completely before appending an assignment."""
        records = self._records[category]
        if possible <= 0 or score < 0 or score > possible:
            raise ValueError("invalid assignment score")
        records.append((score, possible))

    def category_average(self, category):
        """Every assignment has equal weight within its category."""
        records = self._records[category]
        percentages = ((100.0 * score / possible, 1)
                       for score, possible in records)
        return weighted_mean(percentages)

    def overall(self):
        """Only categories with assignments participate in the average."""
        active = []
        for category, weight in self._weights.items():
            average = self.category_average(category)
            if average is not None:
                active.append((average, weight))
        return weighted_mean(active)

    def drop_lowest(self, category, count=1):
        """Select by percentage while retaining insertion order for survivors."""
        records = self._records[category]
        if count < 0:
            raise ValueError("count must be nonnegative")
        ranked = sorted(enumerate(records),
                        key=lambda entry: entry[1][0] / entry[1][1])
        selected = ranked[:count]
        removed_indices = {index for index, record in selected}
        removed = [record for index, record in selected]
        self._records[category] = [
            record for index, record in enumerate(records)
            if index not in removed_indices
        ]
        return removed

    def report(self):
        """Return a detached report, retaining the configured category order."""
        result = {}
        for category in self._weights:
            result[category] = self.category_average(category)
        return result
