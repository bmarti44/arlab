class UnitRegistry:
    def __init__(self):
        self._units = {}

    def register(self, name, dimension, factor, offset=0.0):
        raise NotImplementedError

    def convert(self, value, source, target):
        raise NotImplementedError

    def units(self):
        raise NotImplementedError
