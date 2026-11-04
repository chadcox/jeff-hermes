"""report.py <outdir>: markdown summary of an overnight.sh batch.

Joins <outdir>/results.jsonl with the plugin's shadow log by session id, and applies the
decision rule fixed before the run: the router wins on a model if, against always-think,
its median wall time is <= 0.85x, its total tokens are lower, and the 90% bootstrap
interval of the hidden-test pass fraction difference (router - on, paired by task and rep)
has a lower bound above -0.05. A verdict requires contiguous call records ending in a
final reply and token usage for every call. New runs retain their own telemetry snapshots.
"""
import collections, json, random, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from shadow_report import default_logs, load, load_default

out = Path(sys.argv[1])
rows = [json.loads(l) for l in open(out / "results.jsonl") if l.strip()]
live_sessions = {r["session"] for r in rows if r.get("session") and not r.get("telemetry")}

calls = collections.defaultdict(list)
for log in default_logs() if live_sessions else []:
    for c in load_default(log):
        if c.get("session_id") in live_sessions:
            calls[c["session_id"]].append(c)
for r in rows:
    if r.get("telemetry"):
        path = out / r["telemetry"]
        calls[r.get("session")] = [c for c in load([str(path)]) if c.get("session_id") == r.get("session")] if path.exists() else []


def frac(r):
    return r["passed"] / r["total"] if r.get("total") else 0.0


def full(r):
    return bool(r.get("total")) and r["passed"] == r["total"]


def p90(v):
    v = sorted(v)
    return v[min(len(v) - 1, int(0.9 * len(v)))]


def call_stats(rs):
    cs = [c for r in rs for c in calls.get(r.get("session"), [])]
    cost = [c["main_cost"] if isinstance(c.get("main_cost"), dict) else {} for c in cs]
    guards = [c["guard"] for c in cs if c.get("guard")]
    return {
        "calls": len(cs),
        "off": sum(1 for c in cs if c.get("main_thinking") is False),
        "prompt": sum(x["prompt_tokens"] for x in cost if type(x.get("prompt_tokens")) is int and x["prompt_tokens"] >= 0),
        "completion": sum(x["completion_tokens"] for x in cost if type(x.get("completion_tokens")) is int and x["completion_tokens"] >= 0),
        "reasoning": sum(x.get("reasoning_chars") or 0 for x in cost),
        "main_ms": [x["ms"] for x in cost if x.get("ms") is not None],
        "jeff_ms": [c["router"]["ms"] for c in cs if isinstance(c.get("router"), dict) and c["router"].get("ms") is not None],
        "guards": collections.Counter(
            [f["rule"] for g in guards for f in g.get("forced") or []]
            + [g["trigger"].get("trigger", "?") for g in guards if isinstance(g.get("trigger"), dict)]),
        "triggers": sum(1 for g in guards if g.get("trigger")),
        # Only calls where Jeff's pick decided the level (forced calls think regardless).
        "pxhigh": [c["router"]["probabilities"]["xhigh"] for c in cs
                   if isinstance(c.get("router"), dict) and (c["router"].get("probabilities") or {}).get("xhigh") is not None
                   and not (c.get("guard") or {}).get("forced")],
    }


def boot(diffs, n=10000, seed=0):
    rnd = random.Random(seed)
    m = sorted(st.mean(rnd.choices(diffs, k=len(diffs))) for _ in range(n))
    return m[int(0.05 * n)], m[int(0.95 * n) - 1]


def token_total(rs):
    total = 0
    for r in rs:
        cs = calls.get(r.get("session"), [])
        counts = [c.get("api_call_count") for c in cs]
        if (r.get("timeout") or not cs or any(type(n) is not int for n in counts)
                or sorted(counts) != list(range(1, len(cs) + 1))):
            return None
        last = max(cs, key=lambda c: c["api_call_count"])
        if not isinstance(last.get("main_cost"), dict) or last["main_cost"].get("finish_reason") != "stop":
            return None
        for c in cs:
            cost = c.get("main_cost")
            if not isinstance(cost, dict):
                return None
            tokens = [cost.get("prompt_tokens"), cost.get("completion_tokens")]
            if any(type(n) is not int or n < 0 for n in tokens):
                return None
            total += sum(tokens)
    return total


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
    print("| Mode | Runs | Full passes | Hidden pass frac | Median wall | p90 wall | Timeouts | Tests edited | Calls | Thinking off | Prompt tok | Completion tok | Reasoning chars | Median Main ms/call | Median Jeff ms | Guards | Triggers |")
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
              f" | {dict(s['guards']) or '-'} | {s['triggers']} |")
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
            tok_r = token_total([a for a, _ in pairs])
            tok_o = token_total([b for _, b in pairs])
            if tok_r is None or tok_o is None:
                print(f"\n**Decision ({rm} vs on, {len(pairs)} paired runs): INSUFFICIENT TELEMETRY.** "
                      "Every paired run needs contiguous calls from 1 through a final stop reply, "
                      "with prompt and completion token counts on every call and no timeout.")
                continue
            diffs = [frac(a) - frac(b) for a, b in pairs]
            lo, hi = boot(diffs)
            wall_ratio = st.median(a["wall_s"] for a, _ in pairs) / st.median(b["wall_s"] for _, b in pairs)
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
