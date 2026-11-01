import re
TOK = re.compile(r"\s*(?:(\d+\.?\d*|\.\d+)|(.))")
def evaluate(expr):
    toks = []
    for num, op in TOK.findall(expr):
        if num: toks.append(float(num))
        elif op.strip():
            if op not in "+-*/^()": raise ValueError(op)
            toks.append(op)
    if not toks: raise ValueError("empty")
    pos = [0]
    def peek(): return toks[pos[0]] if pos[0] < len(toks) else None
    def take(): t = peek(); pos[0] += 1; return t
    def expr_():
        v = term()
        while peek() in ("+", "-"):
            v = v + term() if take() == "+" else v - term()
        return v
    def term():
        v = unary()
        while peek() in ("*", "/"):
            if take() == "*": v *= unary()
            else:
                d = unary()
                if d == 0: raise ZeroDivisionError
                v /= d
        return v
    def unary():
        if peek() == "-": take(); return -unary()
        return power()
    def power():
        b = atom()
        if peek() == "^": take(); return b ** unary()
        return b
    def atom():
        t = take()
        if isinstance(t, float): return t
        if t == "(":
            v = expr_()
            if take() != ")": raise ValueError("paren")
            return v
        raise ValueError(t)
    v = expr_()
    if pos[0] != len(toks): raise ValueError("trailing")
    return v
