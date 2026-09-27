def validate_weights(weights, capacity: int) -> list[int]:
    """Materialize and validate a batch before the packer mutates state."""
    if type(capacity) is not int:
        raise TypeError("capacity must be an integer")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    result = []
    for weight in weights:
        if type(weight) is not int:
            raise TypeError("weights must be integers")
        if not 1 <= weight <= capacity:
            raise ValueError("weight is outside bin capacity")
        result.append(weight)
    return result


class BinPacker:
    """Assign items to the earliest bin with sufficient free capacity.

    Bin positions are stable: a newly required bin is always appended.
    Contents and used capacities are maintained together on each insertion.
    """

    def __init__(self, capacity: int):
        validate_weights([], capacity)
        self.capacity = capacity
        self._bins = []
        self._used = []

    def add(self, weight: int) -> int:
        """Add one valid item, returning its stable bin index."""
        validate_weights([weight], self.capacity)
        for index, used in enumerate(self._used):
            if used + weight <= self.capacity:
                self._bins[index].append(weight)
                self._used[index] += weight
                return index
        self._bins.append([weight])
        self._used.append(weight)
        return len(self._bins) - 1

    def add_many(self, weights) -> list[int]:
        """Validate once up front so a bad batch has no partial effects.

        A generator is consumed only during validation. The resulting list
        can then be inserted safely using the same single-item operation.
        """
        checked = validate_weights(weights, self.capacity)
        assignments = []
        for weight in checked:
            assignments.append(self.add(weight))
        return assignments

    def snapshot(self) -> list[list[int]]:
        """Return copies of both the outer list and each bin's contents."""
        return [list(items) for items in self._bins]

    def remaining(self) -> list[int]:
        """Compute a detached view of spare capacity for every bin."""
        free = []
        for used in self._used:
            free.append(self.capacity - used)
        return free


def first_fit(weights, capacity: int) -> list[list[int]]:
    """Pack a complete shipment into a fresh collection of bins.

    This helper has the same ordering and validation rules as incremental
    packing, including validation of capacity for empty shipments.
    """
    packer = BinPacker(capacity)
    packer.add_many(weights)
    return packer.snapshot()
