import math
import pytest
from token_bucket import TokenBucket
from rate_limiter import RateLimiter

def test_bucket_refill_capacity_and_denial():
    b = TokenBucket(5, 2, -1)
    assert (b.capacity, b.refill_rate, b.tokens, b.updated_at) == (5.0, 2.0, 5.0, -1.0)
    assert all(type(v) is float for v in (b.capacity, b.refill_rate, b.tokens, b.updated_at))
    assert b.try_take(4, -1) is True
    assert b.try_take(2, -1) is False
    assert b.tokens == 1
    assert b.advance(-0.5) == 2
    assert b.try_take(2, -0.5) is True
    assert b.advance(100) == 5
    assert b.try_take(6, 101) is False
    assert b.tokens == 5 and b.updated_at == 101

def test_bucket_wait_times_and_zero_refill():
    b = TokenBucket(4, 2, 0)
    assert b.wait_time(4, 0) == 0.0
    b.try_take(4, 0)
    assert b.wait_time(3, 0.5) == 1.0
    assert b.tokens == 1.0
    assert math.isinf(b.wait_time(5, 1))
    z = TokenBucket(2, 0, 0)
    assert z.try_take(2, 0) is True
    assert math.isinf(z.wait_time(1, 100))
    assert z.advance(200) == 0.0

def test_bucket_validation_and_unchanged_state():
    for capacity, rate in [(0, 1), (-1, 1), (True, 1), (2, -1), (2, False), (float('inf'), 1), (2, float('nan'))]:
        with pytest.raises(ValueError):
            TokenBucket(capacity, rate, 0)
    for now in (True, '0', float('nan'), float('inf')):
        with pytest.raises(ValueError):
            TokenBucket(2, 1, now)
    b = TokenBucket(3, 1, 10)
    b.try_take(2, 10)
    for method in (b.try_take, b.wait_time):
        for cost in (0, -1, True, '1', float('inf'), float('nan')):
            with pytest.raises(ValueError):
                method(cost, 20)
            assert (b.tokens, b.updated_at) == (1, 10)
        with pytest.raises(ValueError):
            method(1, 9)
        assert (b.tokens, b.updated_at) == (1, 10)
    for now in (9, True, float('nan')):
        with pytest.raises(ValueError):
            b.advance(now)
        assert (b.tokens, b.updated_at) == (1, 10)

def test_independent_keys_refill_and_default_cost():
    now = [0]
    r = RateLimiter(2, 0.5, lambda: now[0])
    assert r.consume('a') is True
    assert r.consume('a') is True
    assert r.consume('a') is False
    assert r.available('b') == 2.0
    assert r.retry_after('a') == 2.0
    now[0] = 1
    assert r.available('a') == 0.5
    assert r.retry_after('a') == 1.0
    now[0] = 2
    assert r.consume('a') is True
    assert r.available('b') == 2.0
    other = RateLimiter(2, 0.5, lambda: now[0])
    assert other.available('a') == 2.0

def test_atomic_batch_coalesces_duplicate_keys():
    now = [0]
    r = RateLimiter(5, 1, lambda: now[0])
    requests = [('a', 2), ('b', 3), ('a', 1)]
    assert r.consume_many(requests) is True
    assert requests == [('a', 2), ('b', 3), ('a', 1)]
    assert r.available('a') == r.available('b') == 2
    assert r.consume_many([('a', 1), ('b', 3)]) is False
    assert r.available('a') == r.available('b') == 2
    assert r.consume_many([('a', 2), ('a', 1)]) is False
    now[0] = 1
    assert r.consume_many([('a', 4), ('b', 1), ('new', 1)]) is False
    assert r.available('a') == r.available('b') == 3
    assert r.available('new') == 5
    assert r.consume_many([('a', 3), ('b', 3)]) is True
    assert r.available('a') == r.available('b') == 0

def test_one_clock_sample_and_prevalidation():
    calls = []
    def clock():
        calls.append(1)
        return 0
    for capacity, rate in [(0, 1), (2, -1), (True, 1), (2, float('inf'))]:
        with pytest.raises(ValueError):
            RateLimiter(capacity, rate, clock)
    r = RateLimiter(5, 1, clock)
    assert calls == []
    for operation in (lambda: r.consume('a'), lambda: r.retry_after('a'), lambda: r.available('a'), lambda: r.consume_many([('a', 1), ('b', 1)]), lambda: r.consume_many([])):
        before = len(calls)
        operation()
        assert len(calls) == before + 1
    before = len(calls)
    for operation in (lambda: r.consume('', 1), lambda: r.available(1), lambda: r.retry_after('a', True), lambda: r.consume('a', 0), lambda: r.consume_many([('a', 1), ('b', -1)]), lambda: r.consume_many([('a', 1), ('', 1)])):
        with pytest.raises(ValueError):
            operation()
    assert len(calls) == before
    assert r.available('a') == 3
    assert r.consume_many([]) is True

def test_clock_reversal_is_global_and_atomic():
    now = [10]
    r = RateLimiter(4, 1, lambda: now[0])
    assert r.consume('a', 4) is True
    now[0] = 12
    assert r.available('b') == 4
    now[0] = 11
    with pytest.raises(ValueError):
        r.consume('a')
    with pytest.raises(ValueError):
        r.consume_many([('a', 1), ('c', 1)])
    for value in (float('nan'), float('inf'), True, '12'):
        now[0] = value
        with pytest.raises(ValueError):
            r.available('a')
    now[0] = 12
    assert r.available('a') == 2
    assert r.available('b') == 4
    assert r.consume('a', 3) is False
    now[0] = 11
    with pytest.raises(ValueError):
        r.retry_after('a')
    now[0] = 12
    assert r.available('a') == 2

def test_unfulfillable_requests_and_fractional_costs():
    r = RateLimiter(1.5, 0, lambda: -2)
    assert r.consume(' ', 0.5) is True
    assert r.available(' ') == 1.0
    assert r.retry_after(' ', 1) == 0.0
    assert r.consume(' ', 1) is True
    assert math.isinf(r.retry_after(' '))
    assert math.isinf(r.retry_after('new', 2))
    assert r.consume('new', 2) is False
    assert r.available('new') == 1.5
    assert r.consume_many([('fresh', 1), ('fresh', 1)]) is False
    assert r.available('fresh') == 1.5
