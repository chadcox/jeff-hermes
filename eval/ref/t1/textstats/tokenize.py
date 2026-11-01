import re

WORD = re.compile(r"[a-z]+(?:'[a-z]+)*")


def words(text):
    """Lower-cased words. Contractions like "don't" stay one word."""
    return WORD.findall(text.lower())
