"""report.py <outdir>: markdown summary of an overnight.sh batch.

Joins <outdir>/results.jsonl with the plugin's shadow log by session id, and applies the
decision rule fixed before the run: the router wins on a model if, against always-think,
its median wall time is <= 0.85x, its total tokens are lower, and the 90% bootstrap
interval of the hidden-test pass fraction difference (router - on, paired by task and rep)
has a lower bound above -0.05.
"""
import collections, glob, json, random, statistics as st, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = [json.loads(l) for l in open(out / "results.jsonl") if l.strip()]
sessions = {r["session"] for r in rows if r.get("session")}

calls = collections.defaultdict(list)
for f in sorted(glob.glob(str(Path.home() / ".hermes/plugin-data/agent-plugin-jeff_router-*/shadow.jsonl*"))):
    if f.endswith(".lock"):
        continue
    for line in open(f):
        try:
            c = json.loads(line)
        except ValueError:
            continue
        for old in ("strata", "strata_cost", "strata_thinking"):  # older logs used strata* names
            if old in c:
                c.setdefault(old.replace("strata", "main"), c.pop(old))
        if c.get("session_id") in sessions:
            calls[c["session_id"]].append(c)


def frac(r):
    return r["passed"] / r["total"] if r.get("total") else 0.0


def full(r):
    return bool(r.get("total")) and r["passed"] == r["total"]


def p90(v):
    v = sorted(v)
    return v[min(len(v) - 1, int(0.9 * len(v)))]


def call_stats(rs):
    cs = [c for r in rs for c in calls.get(r.get("session"), [])]
    cost = [c.get("main_cost") or {} for c in cs]
    guards = [c["guard"] for c in cs if c.get("guard")]
    return {
        "calls": len(cs),
        "off": sum(1 for c in cs if c.get("main_thinking") is False),
        "prompt": sum(x.get("prompt_tokens") or 0 for x in cost),
        "completion": sum(x.get("completion_tokens") or 0 for x in cost),
        "reasoning": sum(x.get("reasoning_chars") or 0 for x in cost),
        "main_ms": [x["ms"] for x in cost if x.get("ms") is not None],
        "jeff_ms": [c["router"]["ms"] for c in cs if isinstance(c.get("router"), dict) and c["router"].get("ms") is not None],
        "guards": collections.Counter(
            [f["rule"] for g in guards for f in g.get("forced") or []]
            + [g["trigger"].get("trigger", "?") for g in guards if isinstance(g.get("trigger"), dict)]),
        "reasks": sum(1 for g in guards if g.get("trigger")),
        # Only calls where Jeff's pick decided the level (forced calls think regardless).
        "pxhigh": [c["router"]["probabilities"]["xhigh"] for c in cs
                   if isinstance(c.get("router"), dict) and (c["router"].get("probabilities") or {}).get("xhigh") is not None
                   and not (c.get("guard") or {}).get("forced")],
    }


def boot(diffs, n=10000, seed=0):
    rnd = random.Random(seed)
    m = sorted(st.mean(rnd.choices(diffs, k=len(diffs))) for _ in range(n))
    return m[int(0.05 * n)], m[int(0.95 * n) - 1]


by = collections.defaultdict(list)
for r in rows:
    by[(r["model"], r["mode"])].append(r)
models = sorted({r["model"] for r in rows})
modes = sorted({r["mode"] for r in rows}, key=lambda m: ("off", "on").index(m) if m in ("off", "on") else 2)
tasks = sorted({r["task"] for r in rows})

print(f"# jeff_router A/B: {out.name}\n")
print(f"{len(rows)} runs, {len(models)} models, tasks {', '.join(tasks)}.\n")
for model in models:
    print(f"## {model}\n")
    print("| Mode | Runs | Full passes | Hidden pass frac | Median wall | p90 wall | Timeouts | Tests edited | Calls | Thinking off | Prompt tok | Completion tok | Reasoning chars | Median Main ms/call | Median Jeff ms | Guards | Re-asks |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    stats = {}
    for mode in modes:
        rs = by.get((model, mode), [])
        if not rs:
            continue
        s = call_stats(rs)
        stats[mode] = (rs, s)
        walls = [r["wall_s"] for r in rs]
        print(f"| {mode} | {len(rs)} | {sum(map(full, rs))}/{len(rs)} | {st.mean(map(frac, rs)):.3f} | {st.median(walls):.0f}s | {p90(walls):.0f}s"
              f" | {sum(bool(r.get('timeout')) for r in rs)} | {sum(bool(r.get('tests_touched')) for r in rs)} | {s['calls']}"
              f" | {s['off']}/{s['calls']} | {s['prompt']:,} | {s['completion']:,} | {s['reasoning']:,}"
              f" | {st.median(s['main_ms']) if s['main_ms'] else 0:.0f} | {st.median(s['jeff_ms']) if s['jeff_ms'] else 0:.0f}"
              f" | {dict(s['guards']) or '-'} | {s['reasks']} |")
    print("\nPer task (full passes / runs, median wall):\n")
    print("| Task | " + " | ".join(modes) + " |")
    print("|---|" + "---|" * len(modes))
    for t in tasks:
        cells = []
        for mode in modes:
            rs = [r for r in by.get((model, mode), []) if r["task"] == t]
            cells.append(f"{sum(map(full, rs))}/{len(rs)}, {st.median([r['wall_s'] for r in rs]):.0f}s" if rs else "-")
        print(f"| {t} | " + " | ".join(cells) + " |")

    routers = [m for m in modes if m.startswith("router")]
    if "on" in stats:
        on_rs, on_s = stats["on"]
        on_by = {(r["task"], r["rep"]): r for r in on_rs}
        for rm in routers:
            if rm not in stats:
                continue
            rr, rs_ = stats[rm]
            pairs = [(r, on_by[(r["task"], r["rep"])]) for r in rr if (r["task"], r["rep"]) in on_by]
            if not pairs:
                continue
            diffs = [frac(a) - frac(b) for a, b in pairs]
            lo, hi = boot(diffs)
            wall_ratio = st.median(a["wall_s"] for a, _ in pairs) / st.median(b["wall_s"] for _, b in pairs)
            tok_r = sum(c.get("main_cost", {}).get("prompt_tokens", 0) or 0 for a, _ in pairs for c in calls.get(a.get("session"), []))
            tok_r += sum(c.get("main_cost", {}).get("completion_tokens", 0) or 0 for a, _ in pairs for c in calls.get(a.get("session"), []))
            tok_o = sum(c.get("main_cost", {}).get("prompt_tokens", 0) or 0 for _, b in pairs for c in calls.get(b.get("session"), []))
            tok_o += sum(c.get("main_cost", {}).get("completion_tokens", 0) or 0 for _, b in pairs for c in calls.get(b.get("session"), []))
            ok = wall_ratio <= 0.85 and tok_r < tok_o and lo > -0.05
            print(f"\n**Decision ({rm} vs on, {len(pairs)} paired runs):** median wall ratio {wall_ratio:.2f} (need <= 0.85), "
                  f"tokens {tok_r:,} vs {tok_o:,} (need lower), hidden-pass diff {st.mean(diffs):+.3f}, 90% CI [{lo:+.3f}, {hi:+.3f}] (need lower bound > -0.05). "
                  f"**{'ROUTER WINS' if ok else 'ROUTER DOES NOT CLEAR THE BAR'}**")
            px = rs_["pxhigh"]
            if px:
                print(f"\nCounterfactual thresholds over {len(px)} routed calls (share that would think): "
                      + ", ".join(f"{t}: {sum(p >= t for p in px) / len(px):.0%}" for t in (0.2, 0.4, 0.6, 0.8)))
    print()

missing = sum(1 for r in rows if not calls.get(r.get("session")))
if missing:
    print(f"_{missing} runs had no shadow-log records (no session id or not found)._")
