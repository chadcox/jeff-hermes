# wrap.py

Write `wrap.py` with `wrap(text: str, width: int) -> str`.

- Paragraphs are separated by one or more blank lines (lines that are empty or only whitespace). In the output, paragraphs are separated by exactly one empty line.
- Within a paragraph, words are separated by any whitespace including newlines. Fill lines greedily: put as many words on a line as fit within `width` characters, joined by single spaces.
- A word longer than `width` starts on a new line (unless the current line is empty) and is cut into pieces of exactly `width` characters; the last piece may be shorter and following words may join its line.
- No line has leading or trailing spaces. The result has no leading or trailing newlines. Empty or whitespace-only input returns "".
- `width < 1` raises `ValueError`.
