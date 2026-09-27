def validate_weights(weights, capacity: int) -> list[int]:
    """Materialize and validate a batch before the packer mutates state."""
    raise NotImplementedError()

class BinPacker:
    """Assign items to the earliest bin with sufficient free capacity.

    Bin positions are stable: a newly required bin is always appended.
    Contents and used capacities are maintained together on each insertion.
    """

    def __init__(self, capacity: int):
        raise NotImplementedError()

    def add(self, weight: int) -> int:
        """Add one valid item, returning its stable bin index."""
        raise NotImplementedError()

    def add_many(self, weights) -> list[int]:
        """Validate once up front so a bad batch has no partial effects.

        A generator is consumed only during validation. The resulting list
        can then be inserted safely using the same single-item operation.
        """
        raise NotImplementedError()

    def snapshot(self) -> list[list[int]]:
        """Return copies of both the outer list and each bin's contents."""
        raise NotImplementedError()

    def remaining(self) -> list[int]:
        """Compute a detached view of spare capacity for every bin."""
        raise NotImplementedError()

def first_fit(weights, capacity: int) -> list[list[int]]:
    """Pack a complete shipment into a fresh collection of bins.

    This helper has the same ordering and validation rules as incremental
    packing, including validation of capacity for empty shipments.
    """
    raise NotImplementedError()
