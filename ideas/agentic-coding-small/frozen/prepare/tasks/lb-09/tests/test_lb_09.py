import pytest
from cart import Cart
from discount_rules import percentage_discount, best_discount


def test_repeated_adds_accumulate():
    cart = Cart()
    assert cart.add("book", 125, 2) is None
    cart.add("book", 125, 3)
    cart.add("pen", 50)
    assert cart.items() == {"book": (125, 5), "pen": (50, 1)}
    assert cart.subtotal() == 675


def test_threshold_inclusive_and_rounding():
    assert percentage_discount(999, 15, 999) == 149
    assert percentage_discount(998, 15, 999) == 0
    assert percentage_discount(1000, 100, 1000) == 1000
    assert percentage_discount(1000, 0) == 0
    assert percentage_discount(0, 50, 0) == 0


def test_best_single_rule_with_generator():
    assert best_discount(1000, iter([(10, 0), (20, 0), (90, 2000)])) == 200
    assert best_discount(1000, [(20, 0), (20, 0)]) == 200
    assert best_discount(1000, []) == 0


def test_free_shipping_uses_original_subtotal():
    cart = Cart()
    cart.add("item", 1000)
    assert cart.checkout([(20, 0)], shipping=125, free_shipping_at=1000) == {
        "subtotal": 1000, "discount": 200, "shipping": 0, "total": 800}
    assert cart.checkout([(20, 0)], shipping=125, free_shipping_at=1001)["total"] == 925
    assert cart.items() == {"item": (1000, 1)}


def test_combined_cart_flow_and_snapshot():
    cart = Cart()
    cart.add("x", 300)
    cart.add("x", 300, 2)
    cart.add("y", 100)
    snapshot = cart.items()
    snapshot.clear()
    assert cart.remove("x", 1) is None
    assert cart.checkout([(10, 700), (20, 0)], shipping=50, free_shipping_at=700) == {
        "subtotal": 700, "discount": 140, "shipping": 0, "total": 560}
    cart.remove("x", 2)
    cart.remove("y")
    assert cart.items() == {}
    assert cart.checkout(shipping=50) == {"subtotal": 0, "discount": 0, "shipping": 0, "total": 0}


def test_failed_mutations_preserve_accumulated_quantity():
    cart = Cart()
    cart.add("x", 100)
    cart.add("x", 100)
    for action in [lambda: cart.add("x", 101), lambda: cart.add("z", -1),
                   lambda: cart.add("z", True), lambda: cart.add("z", 1, 0),
                   lambda: cart.remove("x", 3), lambda: cart.remove("x", True)]:
        with pytest.raises(ValueError):
            action()
    with pytest.raises(KeyError):
        cart.remove("missing")
    assert cart.items() == {"x": (100, 2)}


def test_rule_and_checkout_validation_preserved():
    assert best_discount(100, [(10, 0), (25, 0)]) == 25
    for args in [(-1, 10, 0), (100, 101, 0), (100, -1, 0),
                 (100, 10, -1), (True, 10, 0), (100, 1.5, 0)]:
        with pytest.raises(ValueError):
            percentage_discount(*args)
    with pytest.raises(ValueError):
        best_discount(100, [(100, 0), (101, 0)])
    with pytest.raises(ValueError):
        best_discount(-1, [])
    cart = Cart()
    with pytest.raises(ValueError):
        cart.checkout([(101, 0)])
    for kwargs in [{"shipping": -1}, {"shipping": True}, {"free_shipping_at": -1},
                   {"free_shipping_at": 1.5}]:
        with pytest.raises(ValueError):
            cart.checkout(**kwargs)
