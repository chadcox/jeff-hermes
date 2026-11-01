import unittest

from textstats.stats import moving_average, sentence_lengths, top_words
from textstats.tokenize import words


class TestTextStats(unittest.TestCase):
    def test_contractions(self):
        self.assertEqual(words("Don't stop, it's fine"), ["don't", "stop", "it's", "fine"])

    def test_top_words_ties_alphabetical(self):
        text = "pear apple pear apple fig fig kiwi"
        self.assertEqual(top_words(text, 3), [("apple", 2), ("fig", 2), ("pear", 2)])

    def test_moving_average(self):
        self.assertEqual(moving_average([1, 2, 3, 4], 2), [1.5, 2.5, 3.5])
        self.assertEqual(moving_average([5], 1), [5.0])

    def test_sentence_lengths(self):
        self.assertEqual(sentence_lengths("Hi there. How are you? Great!"), [2, 3, 1])


if __name__ == "__main__":
    unittest.main()
