from textstats.stats import moving_average, sentence_lengths, top_words
from textstats.tokenize import words
C = [
 lambda: words("Don't stop, it's fine") == ["don't", "stop", "it's", "fine"],
 lambda: words("We'll see; they're OK") == ["we'll", "see", "they're", "ok"],
 lambda: top_words("pear apple pear apple fig fig kiwi", 3) == [("apple", 2), ("fig", 2), ("pear", 2)],
 lambda: top_words("b a c b", 2) == [("b", 2), ("a", 1)],
 lambda: moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5],
 lambda: moving_average([5], 1) == [5.0],
 lambda: moving_average([1, 2], 3) == [],
 lambda: sentence_lengths("Hi there. How are you? Great!") == [2, 3, 1],
 lambda: sentence_lengths("One! Two two? Three three three.") == [1, 2, 3],
]
