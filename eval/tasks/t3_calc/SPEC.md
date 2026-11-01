# calc.py

Write `calc.py` with a function `evaluate(expr: str) -> float`.

- Numbers: integers or decimals (`3`, `2.5`, `.5`).
- Operators: `+ - * /` and `^` (power). `/` is true division.
- Precedence, highest first: `^`, then unary minus, then `* /`, then `+ -`.
- `^` is right-associative: `2^3^2` is `2^9` = 512.
- Unary minus binds looser than `^`: `-2^2` is -4. It may repeat: `--3` is 3. It may follow `^`: `2^-1` is 0.5.
- `* / + -` are left-associative: `8/4/2` is 1.
- Parentheses group. Whitespace anywhere is ignored.
- Raise `ValueError` for malformed input (empty, unbalanced parentheses, dangling operator, unknown characters).
- Raise `ZeroDivisionError` for division by zero.
- Do not use `eval`, `exec`, `ast` or any parsing library.
