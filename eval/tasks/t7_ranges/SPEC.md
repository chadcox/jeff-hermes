# ranges.py

Write `ranges.py` with two functions over integer ranges written in interval notation.

A range is a string `"<open><lo>,<hi><close>"`:

- `<open>` is `[` (includes `lo`) or `(` (excludes `lo`).
- `<close>` is `]` (includes `hi`) or `)` (excludes `hi`).
- `lo` and `hi` are integers, possibly negative. Spaces around the numbers are allowed: `"[ 1 , 5 )"`.
- `lo > hi` is invalid. Anything malformed (other brackets, missing comma, non-integers) is invalid. Invalid ranges raise `ValueError`.
- A range with `lo == hi` is empty unless both ends are inclusive: `"[3,3)"`, `"(3,3]"` and `"(3,3)"` are empty; `"[3,3]"` is the single point 3.

## `merge(ranges)`

Take a list of range strings and return the smallest list of ranges covering exactly the same numbers.

- Treat the ranges as sets of real numbers (not just integers): `"(1,2)"` is not empty, and `"[1,2)"` and `"(2,3]"` do not join because 2 is missing.
- Two ranges join when they overlap, or when they touch at a point that at least one of them includes: `"[1,3)"` + `"[3,5]"` gives `"[1,5]"`.
- When ranges share an endpoint, the merged end is inclusive if any of them includes it.
- Empty ranges are dropped.
- Output strings have no spaces, are sorted by their lower end, and do not overlap or touch.
- Do not modify the input list.

## `contains(ranges, x)`

Return `True` if the number `x` is in any of the ranges, else `False`. Invalid ranges raise `ValueError`.
