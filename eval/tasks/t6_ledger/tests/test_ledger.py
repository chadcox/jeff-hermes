import unittest
from decimal import Decimal

from ledger.parse import parse_amount
from ledger.report import daily_totals, total_usd


class TestLedger(unittest.TestCase):
    def test_parse_plain(self):
        self.assertEqual(parse_amount("1,234.50"), Decimal("1234.50"))

    def test_parse_accounting_negative(self):
        self.assertEqual(parse_amount("(12.00)"), Decimal("-12.00"))

    def test_rounding_matches_statement(self):
        rows = [("2026-01-01", "USD", "0.335")] * 3
        self.assertEqual(total_usd(rows), Decimal("1.02"))

    def test_two_days(self):
        rows = [("2026-01-01", "EUR", "100"), ("2026-01-02", "EUR", "100")]
        self.assertEqual(total_usd(rows), Decimal("222.00"))

    def test_daily_totals_with_refund(self):
        rows = [
            ("2026-01-01", "GBP", "40.00"),
            ("2026-01-02", "GBP", "100.00"),
            ("2026-01-02", "GBP", "(20.00)"),
        ]
        self.assertEqual(
            daily_totals(rows),
            {"2026-01-01": Decimal("50.00"), "2026-01-02": Decimal("101.60")},
        )


if __name__ == "__main__":
    unittest.main()
