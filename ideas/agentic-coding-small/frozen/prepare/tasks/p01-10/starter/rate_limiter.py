from token_bucket import TokenBucket, _number, validate_config, validate_cost

class RateLimiter:

    def __init__(self, capacity, refill_rate, clock):
        raise NotImplementedError()

    def _key(self, key):
        raise NotImplementedError()

    def _sample(self):
        raise NotImplementedError()

    def _bucket(self, key, now):
        raise NotImplementedError()

    def consume(self, key, cost=1):
        raise NotImplementedError()

    def retry_after(self, key, cost=1):
        raise NotImplementedError()

    def available(self, key):
        raise NotImplementedError()

    def consume_many(self, requests):
        raise NotImplementedError()
