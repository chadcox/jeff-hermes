import importlib.util
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shadow_report", REPO / "tools" / "shadow_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def rec(outcome="ok", pick="t1", p=0.9, name="read_file", match=True, read=None, kind="tool", ms=20.0):
    tool = None if outcome != "ok" else {"pick": pick, "p": p, "names": {"t1": name, "t2": "terminal"}}
    return {"outcome": outcome, "jeff_ms": ms, "tool": tool, "read_target": read,
            "tool_match": match if outcome == "ok" else None,
            "read_match": None if read is None else read.pop("_match"), "strata": {"kind": kind}}


def test_load_skips_bad_lines(tmp_path):
    path = tmp_path / "shadow.jsonl"
    path.write_text(json.dumps({"outcome": "ok"}) + "\nnot json\n\n" + json.dumps({"outcome": "busy"}) + "\n")
    assert [r["outcome"] for r in report.load([str(path)])] == ["ok", "busy"]


def test_summary_counts_and_threshold_rows():
    records = [
        rec(p=0.95, match=True),
        rec(p=0.75, match=False),
        rec(p=0.45, pick="t2", name="read_file", match=True),
        rec(outcome="busy"),
        rec(p=0.95, match=True, read={"pick": "r1", "p": 0.9, "_match": True}),
        rec(p=0.95, match=True, read={"pick": "r1", "p": 0.5, "_match": False}),
    ]
    text = report.summarize(records)
    assert "| records | 6 |" in text
    assert "| busy | 1 | 16.7% |" in text
    # t=0.9: 3 of 5 ok records covered, all three comparable and agreeing with Strata; all picks are read_file
    assert "| 0.9 | 60.0% (3) | 100.0% (3) | 60.0% (3) | 100.0% (3) |" in text
    # t=0.7: 4 covered, 4 comparable, 3 agree
    assert "| 0.7 | 80.0% (4) | 75.0% (4) |" in text
    # full read_file call, t=0.5 row: both read records up to t2=0.5 (one match, one mismatch), then only the p=0.9 one
    assert "| 0.5 | 2 / 1 / 1 | 2 / 1 / 1 | 2 / 1 / 1 | 1 / 1 / 0 | 1 / 1 / 0 | 1 / 1 / 0 | 1 / 1 / 0 |" in text
    assert "Strata actions in the log: 6." in text
    assert "accuracy" not in text and "correct" not in text  # Strata is a reference, not ground truth


def test_undecidable_records_count_as_covered_but_not_compared():
    records = [
        rec(p=0.95, match=True),
        rec(p=0.95, match=None),  # tool_match undecidable: covered, not comparable
        rec(p=0.95, match=True, read={"pick": "r1", "p": 0.9, "_match": True}),
        rec(p=0.95, match=True, read={"pick": "r1", "p": 0.9, "_match": None}),  # answered, undecided
        rec(p=0.95, match=True, read={"pick": "none_of_these", "p": 0.9, "_match": False}),  # not a full call
    ]
    text = report.summarize(records)
    assert "| 0.9 | 100.0% (5) | 100.0% (4) |" in text
    assert "| 0.9 | " + " | ".join(["2 / 1 / 0"] * 7) + " |" in text


def test_read_file_pick_against_a_different_strata_action_is_a_mismatch():
    # Jeff would call read_file on r1; Strata answered in text, so read_match is None but tool_match is False
    records = [rec(p=0.95, match=False, kind="text", read={"pick": "r1", "p": 0.9, "_match": None})]
    text = report.summarize(records)
    assert "| 0.9 | " + " | ".join(["1 / 0 / 1"] * 7) + " |" in text


def test_empty_input():
    assert "No records" in report.summarize([])


def test_load_default_rereads_when_rotated_mid_read(tmp_path, monkeypatch):
    log = tmp_path / "shadow.jsonl"
    rotated = tmp_path / "shadow.jsonl.1"
    rotated.write_text(json.dumps({"outcome": "a"}) + "\n")
    log.write_text(json.dumps({"outcome": "b"}) + "\n")
    real_read = report._read
    rotations = []

    def read(path):
        data = real_read(path)
        if path == rotated and not rotations:  # a writer rotates right after the report read <log>.1
            rotations.append(path)
            os.replace(log, rotated)
            log.write_text(json.dumps({"outcome": "c"}) + "\n")
        return data

    monkeypatch.setattr(report, "_read", read)
    # without the re-read this would be ["a", "c"]: "b" lost, and "a" (already rotated out) reported
    assert [r["outcome"] for r in report.load_default(log)] == ["b", "c"]


def test_main_reads_the_namespaced_data_dir_under_hermes_home(tmp_path, monkeypatch, capsys):
    data = tmp_path / "plugin-data" / "agent-plugin-jeff_router-d73ec80e"
    data.mkdir(parents=True)
    (data / "shadow.jsonl.1").write_text(json.dumps(rec()) + "\n")
    (data / "shadow.jsonl").write_text(json.dumps(rec(outcome="busy")) + "\n")
    (tmp_path / "plugin-data" / "agent-plugin-other-12345678").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert report.default_logs() == [data / "shadow.jsonl"]
    assert report.main([]) == 0
    assert "| records | 2 |" in capsys.readouterr().out


def test_main_without_a_data_dir_says_no_records(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert report.main([]) == 1
    assert capsys.readouterr().out.startswith(f"No records in {tmp_path / 'plugin-data'}")


def test_main_with_several_data_dirs_asks_for_a_path(tmp_path, monkeypatch, capsys):
    for digest in ("aaaaaaaa", "bbbbbbbb"):
        data = tmp_path / "plugin-data" / f"agent-plugin-jeff_router-{digest}"
        data.mkdir(parents=True)
        (data / "shadow.jsonl").write_text(json.dumps(rec()) + "\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert report.main([]) == 1
    err = capsys.readouterr().err
    assert "pass the shadow.jsonl" in err
    assert "agent-plugin-jeff_router-aaaaaaaa" in err and "agent-plugin-jeff_router-bbbbbbbb" in err
