from collections import OrderedDict
import math
from cache_entry import Entry

class LRUCache:

    def __init__(self, capacity, clock):
        raise NotImplementedError()

    def _purge(self, now):
        raise NotImplementedError()

    def put(self, key, value, ttl=None):
        raise NotImplementedError()

    def get(self, key, default=None):
        raise NotImplementedError()

    def delete(self, key):
        raise NotImplementedError()

    def keys(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
