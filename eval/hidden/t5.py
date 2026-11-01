from wrap import wrap
def err(*a):
    try: wrap(*a)
    except ValueError: return True
    except Exception: return False
    return False
C = [
 lambda: wrap("the quick brown fox", 10) == "the quick\nbrown fox",
 lambda: wrap("the quick brown fox", 9) == "the quick\nbrown fox",
 lambda: wrap("the quick brown fox", 8) == "the\nquick\nbrown\nfox",
 lambda: wrap("a\nb   c", 80) == "a b c",
 lambda: wrap("one\n\n\n  \ntwo", 80) == "one\n\ntwo",
 lambda: wrap("  \n\nhi\n\n", 5) == "hi",
 lambda: wrap("", 5) == "" and wrap("   \n ", 5) == "",
 lambda: wrap("abcdefghij", 4) == "abcd\nefgh\nij",
 lambda: wrap("xy abcdefghij k", 4) == "xy\nabcd\nefgh\nij k",
 lambda: wrap("abcdefgh", 4) == "abcd\nefgh",
 lambda: wrap("a b", 1) == "a\nb",
 lambda: err("x", 0),
]
