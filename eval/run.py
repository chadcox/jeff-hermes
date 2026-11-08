"""A/B harness for the jeff_router plugin.

  run.py set <mode> <model>                 rewrite ~/.hermes/config.yaml for a mode and model (first backs it up)
  run.py restore                            put the backed-up config.yaml back
  run.py run <mode> <task> <rep> <outdir>   fresh copy of a task, one `hermes chat` run, grade, append to <outdir>/results.jsonl
  run.py unlock <outdir>                    put back the eval folders' permissions if a run was killed while they were locked

Modes: off (never think), guarded (off unless a guard fires), on (always think), router@<threshold>.
A gateway reads plugin settings only at start, so stop the running one after `set` (overnight.sh does).
"""
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL.parent / "tools"))
from shadow_report import default_logs, load_default

CFG = Path.home() / ".hermes/config.yaml"
BACKUP = CFG.with_name("config.yaml.jeff-eval-backup")
HERMES = Path.home() / ".hermes/hermes-agent/venv/bin/hermes"
TIMEOUT = 2400
MODES = {  # every mode sets all of these keys, so no mode inherits another's; other settings are kept
    "router":  {"thinking_router": "true", "main_thinking": "true", "guards": "true", "router_threshold": "{thr}"},
    "on":      {"thinking_router": "false", "main_thinking": "true", "guards": "true"},
    "off":     {"thinking_router": "false", "main_thinking": "false", "guards": "false"},
    "guarded": {"thinking_router": "false", "main_thinking": "false", "guards": "true"},
}
for _keys in MODES.values():
    _keys["log"] = "true"  # telemetry comes from shadow.jsonl, which is off by default
MODE_KEYS = {"thinking_router", "main_thinking", "guards", "router_threshold", "log"}
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
    keys = MODES[name]
    cur = re.search(r"    jeff_router:\n      settings:\n((?:        .*\n)+)", text)
    assert cur, "jeff_router block not found"
    kept = [line for line in cur.group(1).splitlines(keepends=True)
            if line.startswith(" " * 9) or line.strip().split(":")[0] not in MODE_KEYS]
    block = "    jeff_router:\n      settings:\n" + "".join(kept) + "".join(
        f"        {k}: {v.format(thr=thr or '0.6')}\n" for k, v in keys.items())
    text = text[:cur.start()] + block + text[cur.end():]
    text, n = re.subn(r"\Amodel:\n  default: .*\n", f"model:\n  default: {model}\n", text)
    assert n == 1, "model.default not found at the top of config.yaml"
    if not BACKUP.exists():  # keep the user's original, not an earlier `set`
        shutil.copy2(CFG, BACKUP)
    CFG.write_text(text)


def restore():
    if BACKUP.exists():
        shutil.move(str(BACKUP), str(CFG))


def current_model():
    return re.match(r"model:\n  default: (.*)\n", CFG.read_text()).group(1).strip()


def run_name(model, mode, task, rep):
    """One path component per run. Characters outside [A-Za-z0-9._@-] (e.g. the / in org/model) become _, plus a
    short hash of the raw id so org-a/m and org-b/m, or a/b and a_b, never share a name."""
    raw = f"{model}-{mode}-{task}-{rep}"
    safe = re.sub(r"[^A-Za-z0-9._@-]", "_", raw)
    if safe != raw or safe.startswith("."):
        safe = safe.lstrip(".") + "-" + hashlib.sha256(raw.encode()).hexdigest()[:8]
    return safe


def recorded(outdir, model, mode, task, rep):
    try:
        lines = (Path(outdir) / "results.jsonl").read_text().splitlines()
    except FileNotFoundError:
        return False
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if (row.get("model"), row.get("mode"), row.get("task"), row.get("rep")) == (model, mode, task, rep):
            return True
    return False


def code_version():
    """This repo's commit, marked -dirty if plugin or eval code differs from it."""
    git = ["git", "-C", str(EVAL.parent)]
    head = subprocess.run(git + ["rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    if not head:
        return "unknown"
    dirty = subprocess.run(git + ["diff", "--quiet", "HEAD", "--", "plugin", "eval"], capture_output=True).returncode
    return head + ("-dirty" if dirty else "")


def lock_file(outdir):
    return Path(outdir) / "locked-modes.json"


def unlock(outdir):
    """Put back the modes saved before locking. Removes the record once done."""
    saved = lock_file(outdir)
    if not saved.exists():
        return
    for path, mode in json.loads(saved.read_text()).items():
        try:
            os.chmod(path, mode)
        except FileNotFoundError:
            pass
    saved.unlink()


def write_private(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        os.fchmod(f.fileno(), 0o600)
        f.write(text)


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
    if recorded(outdir, model, mode, task, rep):
        sys.exit(f"{model} {mode} {task} {rep} is already in {outdir / 'results.jsonl'}; remove that row first")
    runs = (outdir / "runs").resolve()
    keep = runs / run_name(model, mode, task, rep)
    assert keep.parent == runs, keep  # never rmtree outside runs/
    shutil.rmtree(keep, ignore_errors=True)
    # Work outside this repo, in a git repo of its own, so the agent can't see or touch the
    # task sources, other runs, or this repo's git state. Moved into runs/ after grading.
    d = Path(tempfile.mkdtemp(prefix="jeff-eval-")) / keep.name
    shutil.copytree(EVAL / "tasks" / folder, d)
    subprocess.run("git init -q && git add -A && git -c user.name=eval -c user.email=eval@localhost commit -qm init",
                   shell=True, cwd=d, check=True)
    # The agent runs as this user and has found eval/ before, so hide the hidden checks, reference
    # solutions, task sources and earlier runs' outputs until it is done.
    locked = [x for x in (EVAL / "hidden", EVAL / "ref", EVAL / "tasks", outdir / "runs") if x.is_dir()]
    outdir.mkdir(parents=True, exist_ok=True)
    lock_file(outdir).write_text(json.dumps({str(x): x.stat().st_mode & 0o7777 for x in locked}))
    t0 = time.time()
    try:
        for x in locked:
            x.chmod(0o000)
        p = subprocess.run([str(HERMES), "chat", "-Q", "-q", prompt], cwd=d, capture_output=True, text=True, timeout=TIMEOUT)
        out, timed_out = p.stdout + p.stderr, False
    except subprocess.TimeoutExpired as e:
        out, timed_out = "TIMEOUT " + (e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout or "")), True
    finally:
        unlock(outdir)
    wall = time.time() - t0
    keep.parent.mkdir(parents=True, exist_ok=True)
    write_private(keep.parent / f"{keep.name}.out", out)
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
           "telemetry": str(telemetry), "code": code_version(), **grade}
    with open(outdir / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "set":
        set_mode(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "restore":
        restore()
    elif sys.argv[1] == "unlock":
        unlock(sys.argv[2])
    else:
        run(*sys.argv[2:6])
