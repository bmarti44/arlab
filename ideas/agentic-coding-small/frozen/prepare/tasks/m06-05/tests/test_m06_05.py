import pytest
from order_book import OrderBook


def test_empty_and_non_crossing_orders():
    book = OrderBook()
    assert book.snapshot('buy') == []
    assert book.add('b', 'buy', 99, 2) == []
    assert book.add('s', 'sell', 100, 3) == []
    assert book.snapshot('buy') == [('b', 99, 2)]
    assert book.snapshot('sell') == [('s', 100, 3)]


def test_buy_uses_best_price_then_fifo():
    book = OrderBook()
    for args in [('s1', 'sell', 102, 2), ('s2', 'sell', 100, 2), ('s3', 'sell', 100, 3)]:
        book.add(*args)
    assert book.add('b', 'buy', 102, 6) == [('b', 's2', 100, 2), ('b', 's3', 100, 3), ('b', 's1', 102, 1)]
    assert book.snapshot('sell') == [('s1', 102, 1)]


def test_sell_uses_highest_bid_and_resting_price():
    book = OrderBook()
    book.add('b1', 'buy', 105, 2)
    book.add('b2', 'buy', 110, 1)
    book.add('b3', 'buy', 105, 1)
    assert book.add('s', 'sell', 104, 3) == [('b2', 's', 110, 1), ('b1', 's', 105, 2)]
    assert book.snapshot('buy') == [('b3', 105, 1)]


def test_partial_fill_retains_priority():
    book = OrderBook()
    book.add('a', 'sell', 10, 4)
    book.add('b', 'sell', 10, 4)
    assert book.add('x', 'buy', 10, 2) == [('x', 'a', 10, 2)]
    assert book.add('y', 'buy', 10, 3) == [('y', 'a', 10, 2), ('y', 'b', 10, 1)]
    assert book.snapshot('sell') == [('b', 10, 3)]


def test_remainder_rests_at_incoming_limit():
    book = OrderBook()
    book.add('s', 'sell', 8, 1)
    assert book.add('b', 'buy', 10, 4) == [('b', 's', 8, 1)]
    assert book.snapshot('buy') == [('b', 10, 3)]
    assert book.add('s', 'sell', 9, 1) == [('b', 's', 10, 1)]


def test_cancel_reuse_and_snapshot_isolation():
    book = OrderBook()
    book.add('a', 'buy', 10, 1)
    book.add('b', 'buy', 10, 1)
    assert book.cancel('missing') is False
    assert book.cancel('a') is True
    book.add('a', 'buy', 10, 2)
    snapshot = book.snapshot('buy')
    assert snapshot == [('b', 10, 1), ('a', 10, 2)]
    snapshot.clear()
    assert len(book.snapshot('buy')) == 2


def test_invalid_add_is_atomic():
    book = OrderBook()
    book.add('s', 'sell', 10, 5)
    bad_orders = [('s', 'buy', 11, 2), ('', 'buy', 11, 2), ('b', 'BUY', 11, 2), ('b', 'buy', 0, 2), ('b', 'buy', 11, 0), ('b', 'buy', 11, -2)]
    for args in bad_orders:
        with pytest.raises(ValueError):
            book.add(*args)
        assert book.snapshot('sell') == [('s', 10, 5)]
        assert book.snapshot('buy') == []
    with pytest.raises(ValueError):
        book.snapshot('bad')
