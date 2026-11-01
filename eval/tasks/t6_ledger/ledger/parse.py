"""Parse amounts as they appear in bank exports."""
from decimal import Decimal


def parse_amount(text):
    """Parse '1,234.50', '-12', or accounting negatives like '(12.00)' into a Decimal."""
    s = text.strip().replace(",", "")
    neg = s.startswith("-")
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    value = Decimal(s.lstrip("-"))
    return -value if neg else value
