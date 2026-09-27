from copy import deepcopy


def _merge_value(base_value, override_value):
    """Copy a replacement, or merge a dictionary onto a dictionary.

    A scalar base is discarded when an override introduces a subtree.
    Lists are deliberately handled as opaque replacements.
    """
    if isinstance(override_value, dict):
        if not isinstance(base_value, dict):
            base_value = {}
        return merge_config(base_value, override_value)
    return deepcopy(override_value)


def merge_config(base, override):
    """Recursively apply overrides without sharing mutable input data.

    None is a deletion instruction only when encountered directly as
    a value in an override dictionary. Base data and list contents
    retain their ordinary None values.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if value is None:
            result.pop(key, None)
            continue
        previous = result.get(key)
        result[key] = _merge_value(previous, value)
    return result


class Config:
    """An isolated configuration snapshot with dotted-path access."""

    def __init__(self, data):
        self._data = deepcopy(data)

    def apply(self, override):
        """Create a new snapshot while keeping this one intact."""
        merged = merge_config(self._data, override)
        return Config(merged)

    def get(self, path, default=None):
        """Look up dictionary keys and return an isolated value.

        Validation covers the complete path before traversal, so a
        malformed path raises even when its first key is missing.
        """
        parts = path.split('.')
        if any(part == '' for part in parts):
            raise ValueError('path contains an empty segment')

        current = self._data
        for part in parts:
            if not isinstance(current, dict):
                return deepcopy(default)
            if part not in current:
                return deepcopy(default)
            current = current[part]
        return deepcopy(current)

    def to_dict(self):
        """Export data without exposing internal mutable containers.

        Each call creates a full snapshot, including lists nested
        inside dictionaries and dictionaries nested inside lists.
        """
        return deepcopy(self._data)
