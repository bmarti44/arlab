import math

def _number(value):
    raise NotImplementedError()

def validate_config(capacity, refill_rate):
    raise NotImplementedError()

def validate_cost(cost):
    raise NotImplementedError()

class TokenBucket:

    def __init__(self, capacity, refill_rate, now):
        raise NotImplementedError()

    def advance(self, now):
        raise NotImplementedError()

    def try_take(self, cost, now):
        raise NotImplementedError()

    def wait_time(self, cost, now):
        raise NotImplementedError()
