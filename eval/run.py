"""A/B harness for the jeff_router plugin.

  run.py set <mode> <model>                 rewrite ~/.hermes/config.yaml for a mode and model
  run.py run <mode> <task> <rep> <outdir>   fresh copy of a task, one `hermes chat` run, grade, append to <outdir>/results.jsonl

Modes: off (never think), on (always think), router@<threshold>.
A gateway reads plugin settings only at start, so stop the running one after `set` (overnight.sh does).
"""
import json, re, shutil, subprocess, sys, time
from pathlib import Path

EVAL = Path(__file__).resolve().parent
CFG = Path.home() / ".hermes/config.yaml"
HERMES = Path.home() / ".hermes/hermes-agent/venv/bin/hermes"
TIMEOUT = 2400
MODES = {
    "router": "    jeff_router:\n      settings:\n        thinking_router: true\n        router_threshold: {thr}\n",
    "on":     "    jeff_router:\n      settings:\n        thinking_router: false\n        strata_thinking: true\n",
    "off":    "    jeff_router:\n      settings:\n        thinking_router: false\n        strata_thinking: false\n",
}
FIX = "Run them with 'python3 -m unittest', find the root cause of each failure, and fix the code in {pkg}/. Do not edit the tests and do not hard-code expected values. Re-run until all tests pass, then explain each root cause in one sentence."
SPEC = "Implement SPEC.md in this directory. Write your own tests for the edge cases in the spec and run them until they pass. Reply with a one-paragraph summary."
TASKS = {  # task id -> (folder under tasks/, prompt)
    "t1": ("t1_textstats", "The tests in tests/ are failing. Run them with 'python3 -m unittest', find the cause of each failure, and fix the code in textstats/. Do not edit the tests. Re-run the tests until they all pass, then explain each bug in one sentence."),
    "t2": ("t2_shop", "Two tests in tests/ are failing. " + FIX.format(pkg="shop")),
    "t3": ("t3_calc", SPEC),
    "t4": ("t4_slots", SPEC),
    "t5": ("t5_wrap", SPEC),
    "t6": ("t6_ledger", "Some tests in tests/ are failing. " + FIX.format(pkg="ledger")),
    "t7": ("t7_ranges", SPEC),
    "t8": ("t8_notes", "Rename the function slugfy to slugify everywhere in this project (the definition and every use), then run 'python3 -m unittest' to check the tests pass."),
}


def set_mode(mode, model):
    name, _, thr = mode.partition("@")
    text = CFG.read_text()
    cur = re.search(r"    jeff_router:\n      settings:\n(?:        .*\n)+", text)
    assert cur, "jeff_router block not found"
    text = text[:cur.start()] + MODES[name].format(thr=thr or "0.6") + text[cur.end():]
    text, n = re.subn(r"\Amodel:\n  default: .*\n", f"model:\n  default: {model}\n", text)
    assert n == 1, "model.default not found at the top of config.yaml"
    CFG.write_text(text)


def current_model():
    return re.match(r"model:\n  default: (.*)\n", CFG.read_text()).group(1).strip()


def run(mode, task, rep, outdir):
    folder, prompt = TASKS[task]
    outdir = Path(outdir)
    model = current_model()
    d = outdir / "runs" / f"{model}-{mode}-{task}-{rep}"
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(EVAL / "tasks" / folder, d)
    t0 = time.time()
    try:
        p = subprocess.run([str(HERMES), "chat", "-Q", "-q", prompt], cwd=d, capture_output=True, text=True, timeout=TIMEOUT)
        out, timed_out = p.stdout + p.stderr, False
    except subprocess.TimeoutExpired as e:
        out, timed_out = "TIMEOUT " + (e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout or "")), True
    wall = time.time() - t0
    (d.parent / f"{d.name}.out").write_text(out)
    sid = re.findall(r"session_id:\s*(\S+)", out)
    grade = json.loads(subprocess.run([sys.executable, str(EVAL / "grade.py"), task, str(d)], capture_output=True, text=True).stdout)
    orig_tests = EVAL / "tasks" / folder / "tests"
    tests_touched = orig_tests.exists() and subprocess.run(
        ["diff", "-r", "-x", "__pycache__", str(orig_tests), str(d / "tests")], capture_output=True).returncode != 0
    rec = {"model": model, "mode": mode, "task": task, "rep": rep, "session": sid[-1] if sid else None,
           "wall_s": round(wall, 1), "timeout": timed_out, "tests_touched": tests_touched, **grade}
    with open(outdir / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "set":
        set_mode(sys.argv[2], sys.argv[3])
    else:
        run(*sys.argv[2:6])
