from shop.cart import Cart
from shop.checkout import total
from shop.pricing import unit_price
def _t1():
    Cart().add("apple", 3); return total(Cart().add("pear", 1)) == 1.35
def _t2():
    a = Cart(); a.add("plum", 1); b = Cart(); return b.lines == []
def _t3():
    return total(Cart().add("plum", 1)) == 0.86 and total(Cart().add("plum", 10)) == 7.78 and total(Cart().add("plum", 1)) == 0.86
def _t4():
    return abs(unit_price("apple", 12) - 0.45) < 1e-9 and abs(unit_price("apple", 2) - 0.5) < 1e-9 and abs(unit_price("apple", 10) - 0.45) < 1e-9
def _t5():
    lines = []; c = Cart(lines); c.add("pear", 2); return lines == [("pear", 2)]
def _t6():
    return total(Cart().add("apple", 2).add("pear", 2)) == 3.78
C = [_t1, _t2, _t3, _t4, _t5, _t6]
