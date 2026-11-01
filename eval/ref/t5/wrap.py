import re
def wrap(text, width):
    if width < 1: raise ValueError
    paras = [p.split() for p in re.split(r"\n\s*\n", text)]
    out = []
    for words in paras:
        if not words: continue
        lines, cur = [], ""
        for w in words:
            if len(w) > width:
                if cur: lines.append(cur); cur = ""
                while len(w) > width: lines.append(w[:width]); w = w[width:]
                cur = w; continue
            if not cur: cur = w
            elif len(cur) + 1 + len(w) <= width: cur += " " + w
            else: lines.append(cur); cur = w
        if cur: lines.append(cur)
        out.append("\n".join(lines))
    return "\n\n".join(out)
