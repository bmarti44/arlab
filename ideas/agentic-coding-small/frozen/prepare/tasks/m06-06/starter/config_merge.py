from copy import deepcopy

def _merge_value(base_value, override_value):
    """Copy a replacement, or merge a dictionary onto a dictionary.

    A scalar base is discarded when an override introduces a subtree.
    Lists are deliberately handled as opaque replacements.
    """
    raise NotImplementedError()

def merge_config(base, override):
    """Recursively apply overrides without sharing mutable input data.

    None is a deletion instruction only when encountered directly as
    a value in an override dictionary. Base data and list contents
    retain their ordinary None values.
    """
    raise NotImplementedError()

class Config:
    """An isolated configuration snapshot with dotted-path access."""

    def __init__(self, data):
        raise NotImplementedError()

    def apply(self, override):
        """Create a new snapshot while keeping this one intact."""
        raise NotImplementedError()

    def get(self, path, default=None):
        """Look up dictionary keys and return an isolated value.

        Validation covers the complete path before traversal, so a
        malformed path raises even when its first key is missing.
        """
        raise NotImplementedError()

    def to_dict(self):
        """Export data without exposing internal mutable containers.

        Each call creates a full snapshot, including lists nested
        inside dictionaries and dictionaries nested inside lists.
        """
        raise NotImplementedError()
