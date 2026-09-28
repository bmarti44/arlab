"""PLAN §3.5 statistics and decisions — and nothing else.

A "result" is a dict {"primary": float, "metrics": {...}, "items": {id: score} | None} for one seed.
Comparisons take {seed: result} for candidate A and reference B on the same seeds and split.
"""
from __future__ import annotations

import math
from statistics import mean, stdev

ASSUMED_ITEM_SD = math.sqrt(0.2)  # 20 % discordant items


def sign(direction: str) -> float:
    return 1.0 if direction == "maximize" else -1.0


def sigma(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 and len(set(values)) > 1 else 0.0


def compare(a: dict, b: dict, seeds: list[int], direction: str, sig: float) -> dict:
    """Improvement d of a over b (positive = better) with SE, per §3.5."""
    s = sign(direction)
    n = len(seeds)
    per_seed = [s * (a[k]["primary"] - b[k]["primary"]) for k in seeds]
    if all(a[k].get("items") is not None and b[k].get("items") is not None for k in seeds):
        ids = sorted(a[seeds[0]]["items"])
        for k in seeds:
            if sorted(a[k]["items"]) != ids or sorted(b[k]["items"]) != ids:
                raise ValueError("item ids differ between runs")
        diffs = [s * mean(a[k]["items"][i] - b[k]["items"][i] for k in seeds) for i in ids]
        d = mean(diffs)
        sd = stdev(diffs) if len(diffs) >= 2 else 0.0
        se = math.sqrt(sd**2 / len(ids) + 2 * sig**2 / n)
        return {"d": d, "se": se, "per_seed": per_seed, "n_items": len(ids), "item_sd": sd}
    return {"d": mean(per_seed), "se": math.sqrt(2) * sig / math.sqrt(n), "per_seed": per_seed, "n_items": None}


def expected_holdout_se(sig: float, n_holdout_seeds: int, n_holdout_items: int | None, item_sd: float | None = None) -> float:
    if n_holdout_items is None:
        return math.sqrt(2) * sig / math.sqrt(n_holdout_seeds)
    sd = ASSUMED_ITEM_SD if item_sd is None else item_sd
    return math.sqrt(sd**2 / n_holdout_items + 2 * sig**2 / n_holdout_seeds)


def underpowered(expected_se: float, mes: float) -> bool:
    return 2 * expected_se > mes


def deterministic_enough(sig: float, n_validation_items: int | None) -> bool:
    """Required when seeds.confirm is empty (§3.3 CALIBRATE)."""
    if n_validation_items is None:
        return sig == 0
    return sig <= 0.25 * math.sqrt(0.2 / n_validation_items)


def screen_pass(c: dict, has_confirm: bool) -> bool:
    return c["d"] > (1 if has_confirm else 2) * c["se"]


def confirm_pass(c: dict) -> bool:
    ok = c["d"] > 2 * c["se"]
    if c["n_items"] is None:
        ok = ok and all(x > 0 for x in c["per_seed"])
    return ok


def verdict(c: dict, mes: float, *, is_underpowered: bool, n_experiments: int, stop_reason: str) -> tuple[str, str]:
    """Ordered verdict rules. Returns (verdict, reason/rule)."""
    d, se = c["d"], c["se"]
    seed_ok = c["n_items"] is not None or len(c["per_seed"]) < 2 or all(x > 0 for x in c["per_seed"])
    if d >= mes and d - 2 * se > 0 and seed_ok:
        return "supported", f"d={d:.4g} >= mes={mes:g} and d-2SE={d - 2 * se:.4g} > 0" + ("" if c["n_items"] is not None else " and every holdout seed positive")
    early = stop_reason in ("infra", "disk", "stop", "policy_error")
    if not is_underpowered and n_experiments >= 10 and not early and d + 2 * se < mes:
        return "not_found_at_this_scale", f"holdout upper bound d+2SE={d + 2 * se:.4g} < mes={mes:g} after {n_experiments} experiments"
    if is_underpowered:
        return "inconclusive", "underpowered"
    if n_experiments < 10 or early:
        return "inconclusive", f"stopped_early:{stop_reason}"
    return "inconclusive", "holdout_uncertain"
