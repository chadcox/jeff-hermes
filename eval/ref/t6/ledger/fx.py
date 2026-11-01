"""Daily exchange rates to USD."""
from decimal import Decimal

RATES = {
    ("EUR", "2026-01-01"): Decimal("1.10"),
    ("EUR", "2026-01-02"): Decimal("1.12"),
    ("GBP", "2026-01-01"): Decimal("1.25"),
    ("GBP", "2026-01-02"): Decimal("1.27"),
}

_cache = {}


def rate(currency, day):
    """USD per one unit of `currency` on `day` ('YYYY-MM-DD'). Unknown pairs raise KeyError."""
    if currency == "USD":
        return Decimal(1)
    key = (currency, day)
    if key not in _cache:
        _cache[key] = RATES[key]
    return _cache[key]
