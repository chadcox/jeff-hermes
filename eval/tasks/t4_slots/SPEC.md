# slots.py

Write `slots.py` with `free_slots(busy, day_start, day_end, min_minutes)`.

- `busy` is a list of `(start, end)` pairs of `"HH:MM"` strings, in any order. Busy intervals may overlap or touch.
- `day_start` and `day_end` are `"HH:MM"` strings.
- Return the free gaps inside `[day_start, day_end]` as a list of `(start, end)` `"HH:MM"` pairs, sorted by start, each at least `min_minutes` long.
- A busy interval that starts before `day_start` or ends after `day_end` only blocks the part inside the day. Busy intervals entirely outside the day are ignored.
- An interval with `end <= start` is invalid: raise `ValueError`. Malformed times (not `HH:MM`, hour > 23, minute > 59) also raise `ValueError`.
- Touching intervals (`09:00-10:00` and `10:00-11:00`) leave no gap between them.
