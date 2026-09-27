from token_bucket import TokenBucket, _number, validate_config, validate_cost


class RateLimiter:
    def __init__(self, capacity, refill_rate, clock):
        validate_config(capacity, refill_rate)
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self.clock = clock
        self._buckets = {}
        self._last_time = None

    def _key(self, key):
        if not isinstance(key, str) or not key:
            raise ValueError('invalid key')

    def _sample(self):
        now = self.clock()
        if not _number(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError('invalid clock sample')
        self._last_time = float(now)
        return float(now)

    def _bucket(self, key, now):
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(self.capacity, self.refill_rate, now)
        return self._buckets[key]

    def consume(self, key, cost=1):
        self._key(key)
        validate_cost(cost)
        now = self._sample()
        return self._bucket(key, now).try_take(cost, now)

    def retry_after(self, key, cost=1):
        self._key(key)
        validate_cost(cost)
        now = self._sample()
        return self._bucket(key, now).wait_time(cost, now)

    def available(self, key):
        self._key(key)
        now = self._sample()
        return self._bucket(key, now).advance(now)

    def consume_many(self, requests):
        totals = {}
        for key, cost in requests:
            self._key(key)
            validate_cost(cost)
            totals[key] = totals.get(key, 0.0) + cost
        now = self._sample()
        buckets = {key: self._bucket(key, now) for key in totals}
        for bucket in buckets.values():
            bucket.advance(now)
        if any(buckets[key].tokens < cost for key, cost in totals.items()):
            return False
        for key, cost in totals.items():
            buckets[key].try_take(cost, now)
        return True
