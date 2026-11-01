import unittest

from notes.export import filename
from notes.store import Store
from notes.util import slugify


class TestNotes(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_store(self):
        self.assertEqual(Store().add("My Note", "x"), "my-note")

    def test_filename(self):
        self.assertEqual(filename("A b"), "a-b.md")


if __name__ == "__main__":
    unittest.main()
