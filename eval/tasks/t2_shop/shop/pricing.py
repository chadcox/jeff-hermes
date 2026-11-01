from .catalog import PRICES

BULK_QTY = 10
BULK_DISCOUNT = 0.10

_unit_cache = {}


def unit_price(sku, qty):
    """Price per item. Orders of BULK_QTY or more get BULK_DISCOUNT off."""
    if sku in _unit_cache:
        return _unit_cache[sku]
    price = PRICES[sku]
    if qty >= BULK_QTY:
        price = price * (1 - BULK_DISCOUNT)
    _unit_cache[sku] = price
    return price
