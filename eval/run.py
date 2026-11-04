"""A/B harness for the jeff_router plugin.

  run.py set <mode> <model>                 rewrite ~/.hermes/config.yaml for a mode and model
  run.py run <mode> <task> <rep> <outdir>   fresh copy of a task, one `hermes chat` run, grade, append to <outdir>/results.jsonl

Modes: off (never think), on (always think), router@<threshold>.
A gateway reads plugin settings only at start, so stop the running one after `set` (overnight.sh does).
"""
import json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL.parent / "tools"))
from shadow_report import default_logs, load_default

CFG = Path.home() / ".hermes/config.yaml"
HERMES = Path.home() / ".hermes/hermes-agent/venv/bin/hermes"
TIMEOUT = 2400
MODES = {
    "router": "    jeff_router:\n      settings:\n        thinking_router: true\n        router_threshold: {thr}\n",
    "on":     "    jeff_router:\n      settings:\n        thinking_router: false\n        main_thinking: true\n",
    "off":    "    jeff_router:\n      settings:\n        thinking_router: false\n        main_thinking: false\n",
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


def save_telemetry(session, path):
    records = [c for log in default_logs() for c in load_default(log)
               if session is not None and c.get("session_id") == session]
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        os.fchmod(f.fileno(), 0o600)
        for record in records:
            f.write(json.dumps(record) + "\n")


def run(mode, task, rep, outdir):
    folder, prompt = TASKS[task]
    outdir = Path(outdir)
    model = current_model()
    keep = outdir / "runs" / f"{model}-{mode}-{task}-{rep}"
    shutil.rmtree(keep, ignore_errors=True)
    # Work outside this repo, in a git repo of its own, so the agent can't see or touch the
    # task sources, other runs, or this repo's git state. Moved into runs/ after grading.
    d = Path(tempfile.mkdtemp(prefix="jeff-eval-")) / keep.name
    shutil.copytree(EVAL / "tasks" / folder, d)
    subprocess.run("git init -q && git add -A && git -c user.name=eval -c user.email=eval@localhost commit -qm init",
                   shell=True, cwd=d, check=True)
    t0 = time.time()
    try:
        p = subprocess.run([str(HERMES), "chat", "-Q", "-q", prompt], cwd=d, capture_output=True, text=True, timeout=TIMEOUT)
        out, timed_out = p.stdout + p.stderr, False
    except subprocess.TimeoutExpired as e:
        out, timed_out = "TIMEOUT " + (e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout or "")), True
    wall = time.time() - t0
    keep.parent.mkdir(parents=True, exist_ok=True)
    (keep.parent / f"{keep.name}.out").write_text(out)
    sid = re.findall(r"session_id:\s*(\S+)", out)
    telemetry = Path("runs") / f"{keep.name}.shadow.jsonl"
    save_telemetry(sid[-1] if sid else None, outdir / telemetry)
    grade = json.loads(subprocess.run([sys.executable, str(EVAL / "grade.py"), task, str(d)], capture_output=True, text=True).stdout)
    orig_tests = EVAL / "tasks" / folder / "tests"
    tests_touched = orig_tests.exists() and subprocess.run(
        ["diff", "-r", "-x", "__pycache__", str(orig_tests), str(d / "tests")], capture_output=True).returncode != 0
    shutil.move(str(d), str(keep))
    d.parent.rmdir()
    rec = {"model": model, "mode": mode, "task": task, "rep": rep, "session": sid[-1] if sid else None,
           "wall_s": round(wall, 1), "timeout": timed_out, "tests_touched": tests_touched,
           "telemetry": str(telemetry), **grade}
    with open(outdir / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "set":
        set_mode(sys.argv[2], sys.argv[3])
    else:
        run(*sys.argv[2:6])
