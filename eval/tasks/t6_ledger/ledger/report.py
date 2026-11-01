"""Totals in USD."""
from collections import OrderedDict
from decimal import Decimal, ROUND_HALF_UP

from .fx import rate
from .parse import parse_amount

CENT = Decimal("0.01")


def to_usd(amount, currency, day):
    """One entry in USD, rounded to the cent.

    Each entry is rounded on its own before totals are taken, the way the bank
    statement does it, so totals match the statement to the cent.
    """
    return (amount * rate(currency, day)).quantize(CENT, ROUND_HALF_UP)


def total_usd(rows):
    """rows: iterable of (day, currency, amount_text)."""
    return sum((to_usd(parse_amount(a), c, d) for d, c, a in rows), Decimal("0.00"))


def daily_totals(rows):
    """OrderedDict day -> USD total, days in first-seen order."""
    out = OrderedDict()
    for d, c, a in rows:
        out[d] = out.get(d, Decimal("0.00")) + to_usd(parse_amount(a), c, d)
    return out
