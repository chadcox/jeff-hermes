import re
def _m(t):
    if not isinstance(t, str) or not re.fullmatch(r"\d\d:\d\d", t): raise ValueError(t)
    h, m = int(t[:2]), int(t[3:])
    if h > 23 or m > 59: raise ValueError(t)
    return h * 60 + m
def _s(m): return f"{m//60:02d}:{m%60:02d}"
def free_slots(busy, day_start, day_end, min_minutes):
    ds, de = _m(day_start), _m(day_end)
    iv = []
    for s, e in busy:
        s, e = _m(s), _m(e)
        if e <= s: raise ValueError("bad interval")
        s, e = max(s, ds), min(e, de)
        if s < e: iv.append((s, e))
    iv.sort(); out = []; cur = ds
    for s, e in iv:
        if s > cur and s - cur >= min_minutes: out.append((_s(cur), _s(s)))
        cur = max(cur, e)
    if de > cur and de - cur >= min_minutes: out.append((_s(cur), _s(de)))
    return out
