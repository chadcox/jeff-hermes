import unittest

from shop.cart import Cart
from shop.checkout import total


class TestCheckout(unittest.TestCase):
    def test_single_pear(self):
        Cart().add("apple", 3)
        self.assertEqual(total(Cart().add("pear", 1)), 1.35)

    def test_bulk_discount(self):
        self.assertEqual(total(Cart([]).add("plum", 1)), 0.86)
        self.assertEqual(total(Cart([]).add("plum", 10)), 7.78)

    def test_mixed_order(self):
        self.assertEqual(total(Cart([]).add("apple", 2).add("pear", 2)), 3.78)


if __name__ == "__main__":
    unittest.main()
