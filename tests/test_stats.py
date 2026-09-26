import math

import pytest

from arlab import stats


def res(p, items=None):
    return {"primary": p, "metrics": {}, "items": items}


def test_seed_pack_compare_and_se():
    a = {2: res(0.60), 3: res(0.62)}
    b = {2: res(0.50), 3: res(0.50)}
    c = stats.compare(a, b, [2, 3], "maximize", 0.01)
    assert c["d"] == pytest.approx(0.11)
    assert c["se"] == pytest.approx(math.sqrt(2) * 0.01 / math.sqrt(2))
    assert c["n_items"] is None and stats.confirm_pass(c)


def test_minimize_sign():
    c = stats.compare({1: res(1.0)}, {1: res(1.2)}, [1], "minimize", 0.0)
    assert c["d"] == pytest.approx(0.2)


def test_item_pack_se():
    a = {1: res(0.75, {"a": 1, "b": 1, "c": 1, "d": 0})}
    b = {1: res(0.25, {"a": 0, "b": 0, "c": 1, "d": 0})}
    c = stats.compare(a, b, [1], "maximize", 0.0)
    diffs = [1, 1, 0, 0]
    assert c["d"] == pytest.approx(0.5)
    assert c["se"] == pytest.approx(stats.stdev(diffs) / 2)


def test_confirm_requires_every_seed_positive_for_seed_packs():
    c = {"d": 1.0, "se": 0.1, "per_seed": [2.1, -0.1], "n_items": None}
    assert not stats.confirm_pass(c)


def test_power_and_determinism():
    assert stats.underpowered(0.02, 0.03) and not stats.underpowered(0.01, 0.03)
    assert stats.deterministic_enough(0.0, None) and not stats.deterministic_enough(1e-9, None)
    assert stats.deterministic_enough(0.002, 225) and not stats.deterministic_enough(0.01, 225)


V = dict(is_underpowered=False, n_experiments=20, stop_reason="max_experiments")


def c_(d, se, per=(), items=5):
    return {"d": d, "se": se, "per_seed": list(per), "n_items": items}


def test_verdict_order():
    assert stats.verdict(c_(0.1, 0.01), 0.05, **V)[0] == "supported"
    # large d but interval touches 0 -> not supported, and upper bound >= mes -> holdout_uncertain
    assert stats.verdict(c_(0.1, 0.06), 0.05, **V) == ("inconclusive", "holdout_uncertain")
    assert stats.verdict(c_(0.0, 0.01), 0.05, **V)[0] == "not_found_at_this_scale"
    assert stats.verdict(c_(0.0, 0.01), 0.05, **{**V, "n_experiments": 9}) == ("inconclusive", "stopped_early:max_experiments")
    assert stats.verdict(c_(0.0, 0.01), 0.05, **{**V, "stop_reason": "infra"}) == ("inconclusive", "stopped_early:infra")
    assert stats.verdict(c_(0.0, 0.03), 0.05, **{**V, "is_underpowered": True}) == ("inconclusive", "underpowered")
    # supported wins even when underpowered (allow_underpowered packs)
    assert stats.verdict(c_(0.2, 0.01), 0.05, **{**V, "is_underpowered": True})[0] == "supported"
    # seed packs: every holdout seed must be positive
    assert stats.verdict(c_(0.1, 0.01, per=(0.3, -0.1), items=None), 0.05, **V)[0] != "supported"
    assert stats.verdict(c_(0.1, 0.01, per=(0.1, 0.1), items=None), 0.05, **V)[0] == "supported"
