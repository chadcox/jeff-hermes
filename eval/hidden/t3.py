from calc import evaluate
def ok(e, v): return abs(evaluate(e) - v) < 1e-9
def err(e, ex):
    try: evaluate(e)
    except ex: return True
    except Exception: return False
    return False
C = [
 lambda: ok("1 + 2 * 3", 7), lambda: ok("(1+2)*3", 9), lambda: ok("8/4/2", 1), lambda: ok("10-4-3", 3),
 lambda: ok("2^3^2", 512), lambda: ok("-2^2", -4), lambda: ok("(-2)^2", 4), lambda: ok("--3", 3),
 lambda: ok("2^-1", 0.5), lambda: ok(".5 + 2.5", 3), lambda: ok(" 3 *  -2 ", -6), lambda: ok("2*-3^2", -18),
 lambda: ok("7/2", 3.5), lambda: ok("1-(-1)", 2), lambda: ok("-(2+3)*2", -10),
 lambda: err("", ValueError), lambda: err("(1+2", ValueError), lambda: err("1+2)", ValueError),
 lambda: err("1+", ValueError), lambda: err("2 $ 3", ValueError), lambda: err("1/0", ZeroDivisionError), lambda: err("3 4", ValueError),
]
