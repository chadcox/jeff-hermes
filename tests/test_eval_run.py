"""eval/run.py harness safety: settings kept across mode switches, no duplicate runs, private transcripts, model ids
that can't escape runs/, and folder permissions put back as they were. No GPU, provider or Hermes: all stubbed."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]

CONFIG = """model:
  default: old-model
  provider: local
plugins:
  entries:
    jeff_router:
      settings:
        url: http://10.0.0.5:8765
        timeout_s: 5.0
        thinking_router: true
        router_threshold: 0.6
        guards: true
other: kept
"""


@pytest.fixture
def runner(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("eval_run_under_test", REPO / "eval" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG)
    monkeypatch.setattr(module, "CFG", cfg)
    monkeypatch.setattr(module, "BACKUP", cfg.with_name("config.yaml.jeff-eval-backup"))
    return module


def settings(cfg):
    block = cfg.read_text().split("      settings:\n")[1]
    lines = [line.strip() for line in block.splitlines() if line.startswith(" " * 8)]
    return dict(line.split(": ", 1) for line in lines)


def test_mode_switch_keeps_custom_keys_and_sets_every_mode_key(runner):
    expected = {
        "off": {"thinking_router": "false", "main_thinking": "false", "guards": "false"},
        "on": {"thinking_router": "false", "main_thinking": "true", "guards": "true"},
        "guarded": {"thinking_router": "false", "main_thinking": "false", "guards": "true"},
        "router@0.7": {"thinking_router": "true", "main_thinking": "true", "guards": "true", "router_threshold": "0.7"},
    }
    for mode in ("off", "router@0.7", "on", "guarded", "router@0.7", "off"):
        runner.set_mode(mode, "new-model")
        got = settings(runner.CFG)
        assert got.pop("url") == "http://10.0.0.5:8765" and got.pop("timeout_s") == "5.0"
        assert got == {**expected[mode], "log": "true"}  # nothing left over; the log is on for telemetry
    text = runner.CFG.read_text()
    assert text.startswith("model:\n  default: new-model\n") and text.endswith("other: kept\n")
    assert runner.BACKUP.read_text() == CONFIG


def test_mode_switch_keeps_nested_settings(runner):
    runner.CFG.write_text(CONFIG.replace("        guards: true\n", "        guards: true\n        extra:\n          a: 1\n"))
    runner.set_mode("off", "m")
    assert "        extra:\n          a: 1\n" in runner.CFG.read_text()


@pytest.mark.parametrize("mode, text", [("nonsense", CONFIG), ("off", CONFIG.replace("jeff_router", "other_plugin")),
                                        ("off", CONFIG.replace("model:\n  default", "model:\n  name"))])
def test_bad_input_leaves_the_config_intact(runner, mode, text):
    runner.CFG.write_text(text)
    with pytest.raises((KeyError, AssertionError)):
        runner.set_mode(mode, "m")
    assert runner.CFG.read_text() == text


def test_run_name_keeps_plain_ids_and_separates_unsafe_ones(runner):
    assert runner.run_name("Qwen3.8-27B-UD-IQ3_S-5080-MTP", "router@0.6", "t1", "r1") == \
        "Qwen3.8-27B-UD-IQ3_S-5080-MTP-router@0.6-t1-r1"  # existing batches resume under the same names
    names = {runner.run_name(m, "off", "t1", "r1") for m in ("orgA/model", "orgB/model", "a/b", "a_b")}
    assert len(names) == 4 and not any("/" in n for n in names)
    for model in ("../../etc", "..", "/abs/path", ".hidden"):
        name = runner.run_name(model, "off", "t1", "r1")
        assert "/" not in name and not name.startswith(".")


@pytest.fixture
def harness(runner, tmp_path, monkeypatch):
    """A fake eval/ tree and a stubbed `hermes chat` and grader."""
    ev = tmp_path / "eval"
    (ev / "tasks" / "t3_calc" / "tests").mkdir(parents=True)
    (ev / "tasks" / "t3_calc" / "SPEC.md").write_text("spec\n")
    for name, mode in (("hidden", 0o700), ("ref", 0o750)):
        (ev / name).mkdir()
        (ev / name).chmod(mode)
    (ev / "tasks").chmod(0o711)
    monkeypatch.setattr(runner, "EVAL", ev)
    monkeypatch.setattr(runner, "default_logs", lambda: [])
    monkeypatch.setattr(runner, "current_model", lambda: "org/model")
    seen = {}
    real_run = subprocess.run

    def fake_run(args, **kw):
        if isinstance(args, list) and args[0] == str(runner.HERMES):
            seen["modes"] = {p.name: p.stat().st_mode & 0o777 for p in (ev / "hidden", ev / "ref", ev / "tasks")}
            if seen.get("fail"):
                raise RuntimeError("hermes crashed")
            return SimpleNamespace(stdout="session_id: s1\n", stderr="")
        if isinstance(args, list) and args[1] == str(ev / "grade.py"):
            return SimpleNamespace(stdout='{"passed": 1, "total": 1}')
        return real_run(args, **kw)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    return runner, ev, tmp_path / "out", seen


def test_run_locks_then_restores_original_permissions(harness):
    runner, ev, out, seen = harness
    runner.run("off", "t3", "r1", out)
    assert seen["modes"] == {"hidden": 0, "ref": 0, "tasks": 0}
    assert {p.name: p.stat().st_mode & 0o777 for p in (ev / "hidden", ev / "ref", ev / "tasks")} == \
        {"hidden": 0o700, "ref": 0o750, "tasks": 0o711}
    assert not runner.lock_file(out).exists()


def test_unlock_restores_permissions_after_a_killed_run(harness):
    runner, ev, out, seen = harness
    out.mkdir()
    runner.lock_file(out).write_text(json.dumps({str(ev / "hidden"): 0o700, str(ev / "ref"): 0o750}))
    (ev / "hidden").chmod(0)
    (ev / "ref").chmod(0)
    runner.unlock(out)
    assert (ev / "hidden").stat().st_mode & 0o777 == 0o700 and (ev / "ref").stat().st_mode & 0o777 == 0o750
    runner.unlock(out)  # nothing saved: a no-op


def test_failed_run_still_restores_permissions(harness):
    runner, ev, out, seen = harness
    seen["fail"] = True
    with pytest.raises(RuntimeError):
        runner.run("off", "t3", "r1", out)
    assert (ev / "hidden").stat().st_mode & 0o777 == 0o700


def test_run_writes_private_transcript_under_a_safe_name(harness):
    runner, ev, out, seen = harness
    runner.run("off", "t3", "r1", out)
    [row] = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert row["model"] == "org/model" and row["code"] == "unknown"  # the fake eval tree is not a git checkout
    entries = list((out / "runs").iterdir())
    assert all(e.parent == out / "runs" for e in entries) and not (out / "runs" / "org").exists()
    [transcript] = [e for e in entries if e.suffix == ".out"]
    assert transcript.stat().st_mode & 0o777 == 0o600
    assert (out / row["telemetry"]).parent == out / "runs"


def test_duplicate_run_is_refused_and_keeps_the_first(harness):
    runner, ev, out, seen = harness
    runner.run("off", "t3", "r1", out)
    workspace = [e for e in (out / "runs").iterdir() if e.is_dir()][0]
    (workspace / "marker").write_text("first")
    with pytest.raises(SystemExit, match="already in"):
        runner.run("off", "t3", "r1", out)
    assert len((out / "results.jsonl").read_text().splitlines()) == 1 and (workspace / "marker").exists()
    runner.run("on", "t3", "r1", out)  # a different mode is a different run
    assert len((out / "results.jsonl").read_text().splitlines()) == 2


def test_overnight_stops_the_batch_when_a_run_fails():
    text = (REPO / "eval" / "overnight.sh").read_text()
    assert 'run.py" run "$mode" "$task" "r$rep" "$OUT" \\\n          || { echo "ABORT run.py run failed' in text
    assert 'run.py" unlock "$OUT"' in text and "chmod 755" not in text
