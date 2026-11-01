"""grade.py <task> <dir>: run hidden checks in a subprocess; prints passed/total."""
import json, subprocess, sys
from pathlib import Path
HIDDEN = Path(__file__).resolve().parent / "hidden"
task, d = sys.argv[1], sys.argv[2]
code = f"""
import sys, json, signal
sys.path.insert(0, {d!r})
signal.alarm(20)
res = []
try:
    exec(open({str(HIDDEN / (task + '.py'))!r}).read())
    for c in C:
        try: res.append(bool(c()))
        except Exception: res.append(False)
except Exception as e:
    print(json.dumps({{"import_error": repr(e)[:200]}})); sys.exit()
print(json.dumps({{"results": res}}))
"""
try:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, cwd=d).stdout.strip().splitlines()
    r: dict = json.loads(out[-1]) if out else {"import_error": "no output"}
except Exception as e:
    r = {"import_error": repr(e)}
res = r.get("results", [])
print(json.dumps({"passed": sum(res), "total": len(res), "fails": [i for i, x in enumerate(res) if not x], **({"error": r["import_error"]} if "import_error" in r else {})}))
