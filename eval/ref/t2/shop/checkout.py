from .pricing import unit_price

TAX_RATE = 0.08


def total(cart):
    """Order total including tax, rounded to cents."""
    subtotal = 0.0
    for sku, qty in cart.lines:
        subtotal += unit_price(sku, qty) * qty
    # FIXME: is float rounding safe here?
    return round(subtotal * (1 + TAX_RATE), 2)
