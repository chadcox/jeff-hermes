from decimal import Decimal
from ledger.parse import parse_amount
from ledger.fx import rate
from ledger.report import to_usd, total_usd, daily_totals
def _t1():
    return parse_amount("(1,000.25)") == Decimal("-1000.25") and parse_amount(" -5 ") == Decimal("-5") and parse_amount("7") == Decimal("7")
def _t2():
    a = rate("GBP", "2026-01-01"); b = rate("GBP", "2026-01-02"); c = rate("GBP", "2026-01-01")
    return (a, b, c) == (Decimal("1.25"), Decimal("1.27"), Decimal("1.25"))
def _t3():
    return rate("EUR", "2026-01-02") == Decimal("1.12") and rate("EUR", "2026-01-01") == Decimal("1.10")
def _t4():
    try: rate("JPY", "2026-01-01")
    except KeyError: return True
    return False
def _t5():
    return total_usd([("2026-01-02", "USD", "0.335")] * 3) == Decimal("1.02") and to_usd(Decimal("0.125"), "USD", "2026-01-01") == Decimal("0.13")
def _t6():
    rows = [("2026-01-02", "EUR", "10.005"), ("2026-01-01", "EUR", "(3)"), ("2026-01-02", "GBP", "1,000")]
    d = daily_totals(rows)
    return list(d) == ["2026-01-02", "2026-01-01"] and d["2026-01-02"] == Decimal("1281.21") and d["2026-01-01"] == Decimal("-3.30")
def _t7():
    return total_usd([("2026-01-02", "EUR", "100"), ("2026-01-01", "EUR", "100"), ("2026-01-02", "GBP", "(10)")]) == Decimal("209.30")
C = [_t1, _t2, _t3, _t4, _t5, _t6, _t7]
