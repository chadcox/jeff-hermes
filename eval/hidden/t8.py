import pathlib
from notes.util import slugify
from notes.store import Store
from notes.export import filename
def _t1(): return slugify("  Two  Words ") == "two-words"
def _t2(): return Store().add("X Y", "b") == "x-y" and filename("Q&A") == "q-a.md"
def _t3(): return not any("slugfy" in p.read_text() for p in pathlib.Path("notes").glob("*.py"))
C = [_t1, _t2, _t3]
