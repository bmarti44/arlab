import math


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_config(capacity, refill_rate):
    if not _number(capacity) or capacity <= 0 or not _number(refill_rate) or refill_rate < 0:
        raise ValueError('invalid configuration')


def validate_cost(cost):
    if not _number(cost) or cost <= 0:
        raise ValueError('invalid cost')


class TokenBucket:
    def __init__(self, capacity, refill_rate, now):
        validate_config(capacity, refill_rate)
        if not _number(now):
            raise ValueError('invalid time')
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self.tokens = float(capacity)
        self.updated_at = float(now)

    def advance(self, now):
        if not _number(now) or now < self.updated_at:
            raise ValueError('invalid time')
        self.tokens = min(self.capacity, self.tokens + (now - self.updated_at) * self.refill_rate)
        self.updated_at = float(now)
        return self.tokens

    def try_take(self, cost, now):
        validate_cost(cost)
        self.advance(now)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def wait_time(self, cost, now):
        validate_cost(cost)
        self.advance(now)
        if self.tokens >= cost:
            return 0.0
        if cost > self.capacity or self.refill_rate == 0:
            return math.inf
        return (cost - self.tokens) / self.refill_rate
