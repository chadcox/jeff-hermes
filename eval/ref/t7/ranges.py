import re

_RE = re.compile(r"^([\[(])\s*(-?\d+)\s*,\s*(-?\d+)\s*([\])])$")


def _parse(s):
    m = _RE.match(s.strip()) if isinstance(s, str) else None
    if not m:
        raise ValueError(f"bad range: {s!r}")
    lo, hi = int(m.group(2)), int(m.group(3))
    if lo > hi:
        raise ValueError(f"lo > hi: {s!r}")
    return lo, m.group(1) == "[", hi, m.group(4) == "]"


def _empty(r):
    lo, li, hi, hi_inc = r
    return lo == hi and not (li and hi_inc)


def _fmt(r):
    lo, li, hi, hi_inc = r
    return f"{'[' if li else '('}{lo},{hi}{']' if hi_inc else ')'}"


def merge(ranges):
    rs = sorted((r for r in map(_parse, ranges) if not _empty(r)), key=lambda r: (r[0], not r[1]))
    out = []
    for lo, li, hi, hi_inc in rs:
        if out:
            plo, pli, phi, phi_inc = out[-1]
            if lo < phi or (lo == phi and (li or phi_inc)):
                if hi > phi:
                    out[-1] = (plo, pli, hi, hi_inc)
                elif hi == phi:
                    out[-1] = (plo, pli, phi, phi_inc or hi_inc)
                continue
        out.append((lo, li, hi, hi_inc))
    return [_fmt(r) for r in out]


def contains(ranges, x):
    for lo, li, hi, hi_inc in map(_parse, ranges):
        if (lo < x or (li and lo == x)) and (x < hi or (hi_inc and x == hi)):
            return True
    return False
