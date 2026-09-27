class Humanizer:
    def __init__(self, now):
        raise NotImplementedError

    def seconds_from(self, moment):
        raise NotImplementedError

    def describe(self, moment):
        raise NotImplementedError

    def describe_many(self, moments):
        raise NotImplementedError
