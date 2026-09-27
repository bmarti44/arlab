from fractions import Fraction
from units import normalize_unit, parse_amount


def scale_recipe(ingredients, servings, target_servings) -> list[dict]:
    """Scale once, normalize units, and merge matching ingredient groups."""
    original = parse_amount(servings)
    target = parse_amount(target_servings)
    if original <= 0 or target <= 0:
        raise ValueError("servings must be positive")
    ratio = target / original
    totals = {}
    for ingredient in ingredients:
        name = ingredient["name"].strip().lower()
        if not name:
            raise ValueError("empty ingredient name")
        amount = parse_amount(ingredient["amount"])
        unit, factor = normalize_unit(ingredient["unit"])
        key = (name, unit)
        if key not in totals:
            totals[key] = Fraction(0)
        canonical_amount = amount * factor
        scaled_amount = canonical_amount * ratio
        totals[key] += scaled_amount
    result = []
    for (name, unit), total in totals.items():
        rounded_amount = float(round(total, 6))
        result.append({
            "name": name,
            "amount": rounded_amount,
            "unit": unit,
        })
    return result
