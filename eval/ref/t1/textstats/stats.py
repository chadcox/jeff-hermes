import re
from collections import Counter

from .tokenize import words


def top_words(text, n):
    """The n most common words, most common first; ties in alphabetical order."""
    counts = Counter(words(text))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:n]


def moving_average(values, window):
    """Average of each full window of `window` consecutive values."""
    out = []
    for i in range(len(values) - window + 1):
        chunk = values[i:i + window]
        out.append(sum(chunk) / window)
    return out


def sentence_lengths(text):
    """Word count of each sentence. Sentences end with . ! or ?"""
    parts = re.split(r"[.!?]", text)
    return [len(words(p)) for p in parts if p.strip()]
