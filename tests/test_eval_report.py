import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EVAL = Path(__file__).resolve().parents[1] / "eval"


def record(session, count=1, finish="stop", tokens=50):
    return {"session_id": session, "api_call_count": count,
            "main_cost": {"finish_reason": finish, "prompt_tokens": tokens, "completion_tokens": tokens}}


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def batch(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    out = tmp_path / "batch"
    rows = [{"model": "m", "mode": mode, "task": "t1", "rep": "r1", "session": session,
             "wall_s": wall, "passed": 1, "total": 1}
            for mode, session, wall in [("on", "baseline", 100), ("router@0.6", "router", 50)]]
    write_records(out / "results.jsonl", rows)
    log = tmp_path / "hermes/plugin-data/agent-plugin-jeff_router-test/shadow.jsonl"
    write_records(log, [record("baseline", tokens=100), record("router")])
    return out, rows, log


def report(out):
    result = subprocess.run([sys.executable, str(EVAL / "report.py"), str(out)],
                            capture_output=True, text=True, check=True)
    return result.stdout


def test_complete_telemetry_can_win_or_lose(batch):
    out, _, log = batch
    assert "ROUTER WINS" in report(out)
    write_records(log, [record("baseline", tokens=100), record("router", tokens=150)])
    assert "ROUTER DOES NOT CLEAR THE BAR" in report(out)


@pytest.mark.parametrize("missing", ["baseline", "router"])
def test_missing_session_cannot_win(batch, missing):
    out, _, log = batch
    write_records(log, [record(s) for s in ("baseline", "router") if s != missing])
    text = report(out)
    assert "INSUFFICIENT TELEMETRY" in text and "ROUTER WINS" not in text


@pytest.mark.parametrize("cost", [None, {}, "invalid", {"finish_reason": "stop", "prompt_tokens": 10},
                                 {"finish_reason": "stop", "prompt_tokens": None, "completion_tokens": 10},
                                 {"finish_reason": "stop", "prompt_tokens": -1, "completion_tokens": 10},
                                 {"finish_reason": "stop", "prompt_tokens": "10", "completion_tokens": 10},
                                 {"finish_reason": "stop", "prompt_tokens": True, "completion_tokens": 10}])
def test_unusable_token_usage_cannot_win_or_crash(batch, cost):
    out, _, log = batch
    bad = {**record("router"), "main_cost": cost}
    write_records(log, [record("baseline", tokens=100), bad])
    text = report(out)
    assert "INSUFFICIENT TELEMETRY" in text and "ROUTER WINS" not in text


@pytest.mark.parametrize("calls", [[], [record("router", count=2)],
                                  [record("router", finish="tool_calls"), record("router", count=3)],
                                  [record("router"), record("router")],
                                  [record("router", finish="tool_calls")]])
def test_incomplete_call_sequence_cannot_win(batch, calls):
    out, _, log = batch
    write_records(log, [record("baseline", tokens=100), *calls])
    assert "INSUFFICIENT TELEMETRY" in report(out)


def test_out_of_order_calls_and_zero_usage_are_valid(batch):
    out, _, log = batch
    write_records(log, [record("baseline", tokens=100), record("router", count=2, tokens=0),
                        record("router", finish="tool_calls", tokens=10)])
    assert "ROUTER WINS" in report(out)


def test_timeout_cannot_win_when_no_pair_is_left(batch):
    out, rows, _ = batch
    rows[1]["timeout"] = True
    write_records(out / "results.jsonl", rows)
    assert "INSUFFICIENT TELEMETRY" in report(out)


def test_snapshots_survive_live_log_rotation_and_missing_archive_is_not_hidden(batch):
    out, rows, log = batch
    for r in rows:
        r["telemetry"] = f"runs/{r['session']}.shadow.jsonl"
        write_records(out / r["telemetry"], [record(r["session"], tokens=100 if r["mode"] == "on" else 50)])
    write_records(out / "results.jsonl", rows)
    log.unlink()
    assert "ROUTER WINS" in report(out)
    write_records(log, [record("baseline", tokens=100), record("router")])
    (out / rows[1]["telemetry"]).unlink()
    assert "INSUFFICIENT TELEMETRY" in report(out)


def test_run_archives_only_its_session_with_private_permissions(batch, monkeypatch, capsys):
    out, _, log = batch
    write_records(log.with_name(log.name + ".1"), [record("router", finish="tool_calls")])
    write_records(log, [record("router", count=2), record("unrelated")])
    spec = importlib.util.spec_from_file_location("eval_run", EVAL / "run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "current_model", lambda: "m")
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **kw: str(out / "temp"))

    def fake_run(args, **kw):
        if isinstance(args, list) and args[0] == str(runner.HERMES):
            return SimpleNamespace(stdout="session_id: router\n", stderr="")
        if isinstance(args, list) and args[1] == str(EVAL / "grade.py"):
            return SimpleNamespace(stdout='{"passed": 1, "total": 1}')
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.run("router@0.6", "t3", "r1", out)
    row = json.loads(capsys.readouterr().out)
    archive = out / row["telemetry"]
    saved = [json.loads(line) for line in archive.read_text().splitlines()]
    assert [c["api_call_count"] for c in saved] == [1, 2]
    assert all(c["session_id"] == "router" for c in saved)
    assert archive.stat().st_mode & 0o777 == 0o600


def multi_task_batch(out, log, router_passes):
    """Two tasks x four reps, on vs router; `router_passes(task, rep)` says whether the router run fully passed."""
    rows, recs = [], []
    for task in ("t1", "t2"):
        for rep in range(4):
            for mode in ("on", "router@0.6"):
                session = f"{mode}-{task}-{rep}"
                ok = True if mode == "on" else router_passes(task, rep)
                rows.append({"model": "m", "mode": mode, "task": task, "rep": f"r{rep}", "session": session,
                             "wall_s": 100 if mode == "on" else 50, "passed": 2 if ok else 0, "total": 2,
                             "code": "abc1234"})
                recs.append(record(session, tokens=100 if mode == "on" else 50))
    write_records(out / "results.jsonl", rows)
    write_records(log, recs)


def decision(text):
    return next(line for line in text.splitlines() if line.startswith("**Decision"))


def test_bootstrap_resamples_tasks_not_reps(batch):
    out, _, log = batch
    multi_task_batch(out, log, lambda task, rep: task == "t1")  # t2 always fails under the router
    line = decision(report(out))
    assert "hidden-pass diff -0.500" in line
    # every rep of a task moves together, so resampling tasks spans the whole range rather than shrinking around -0.5
    assert "90% CI [-1.000, +0.000]" in line and "ROUTER DOES NOT CLEAR THE BAR" in line


def test_report_shows_paired_outcomes_by_task(batch):
    out, _, log = batch
    multi_task_batch(out, log, lambda task, rep: not (task == "t2" and rep < 3))
    text = report(out)
    assert "Paired full passes (router@0.6 / on): both 5, router@0.6 only 0, on only 3, neither 0." in text
    assert "Discordant by task (router@0.6 only / on only): t2 0/3." in text


def test_report_shows_total_tokens_passing_latency_and_code_version(batch):
    out, _, log = batch
    multi_task_batch(out, log, lambda task, rep: rep != 0)
    text = report(out)
    assert "Code versions: abc1234 (16 runs)." in text
    assert "| Total tok |" in text and "| Median wall, full passes |" in text
    row = next(line for line in text.splitlines() if line.startswith("| router@0.6 |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[5] == "50s" and cells[13] == "800"  # 6 passing runs at 50s; 8 runs x 100 tokens


def test_report_without_full_passes_shows_a_dash_and_unrecorded_code(batch):
    out, rows, _ = batch
    write_records(out / "results.jsonl", [{**r, "passed": 0} for r in rows])
    text = report(out)
    assert "Code versions: unrecorded (2 runs)." in text
    row = next(line for line in text.splitlines() if line.startswith("| on |"))
    assert [c.strip() for c in row.strip("|").split("|")][5] == "-"


def test_timed_out_pair_is_left_out_of_tokens_only(batch):
    out, _, log = batch
    multi_task_batch(out, log, lambda task, rep: True)
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    slow = next(r for r in rows if r["mode"] == "on" and r["task"] == "t1" and r["rep"] == "r0")
    slow["timeout"] = True
    write_records(out / "results.jsonl", rows)
    # the timed-out run never logged a final stop reply
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    write_records(log, [r for r in recs if r["session_id"] != slow["session"]] + [record(slow["session"], finish="length")])
    line = decision(report(out))
    assert "8 paired runs" in line
    assert "tokens 700 vs 1,400 (need lower; 1 timed-out pair left out)" in line
    assert "ROUTER WINS" in line
