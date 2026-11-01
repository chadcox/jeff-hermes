import json
import subprocess
import sys
import threading
from pathlib import Path

from jeff_router_under_test.shadow_log import ShadowLog

MODULE = Path(__file__).resolve().parents[1] / "plugin" / "jeff_router" / "shadow_log.py"
WORKER = """
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("shadow_log_worker", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
log = module.ShadowLog(pathlib.Path(sys.argv[2]), int(sys.argv[3]))
for k in range(200):
    log.append({"p": int(sys.argv[4]), "k": k, "pad": "y" * 100})
"""


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_append_creates_dirs_and_writes_one_line_per_record(tmp_path):
    log = ShadowLog(tmp_path / "deep" / "shadow.jsonl", max_bytes=10_000)
    log.append({"a": 1})
    log.append({"b": "ü"})
    assert read_lines(tmp_path / "deep" / "shadow.jsonl") == [{"a": 1}, {"b": "ü"}]
    assert (tmp_path / "deep" / "shadow.jsonl").stat().st_mode & 0o777 == 0o600


def test_pre_existing_files_are_made_owner_only(tmp_path):
    path = tmp_path / "shadow.jsonl"
    lock = tmp_path / "shadow.jsonl.lock"
    for f in (path, lock):
        f.touch()
        f.chmod(0o644)
    ShadowLog(path, max_bytes=10_000).append({"a": 1})
    assert path.stat().st_mode & 0o777 == 0o600
    assert lock.stat().st_mode & 0o777 == 0o600


def test_fresh_parent_directory_is_owner_only(tmp_path):
    parent = tmp_path / "outer" / "logs"
    ShadowLog(parent / "shadow.jsonl", max_bytes=10_000).append({"a": 1})
    assert parent.stat().st_mode & 0o777 == 0o700


def test_rotation_keeps_one_previous_file(tmp_path):
    path = tmp_path / "shadow.jsonl"
    log = ShadowLog(path, max_bytes=60)
    for i in range(6):
        log.append({"i": i, "pad": "x" * 10})
    rotated = tmp_path / "shadow.jsonl.1"
    assert rotated.exists()
    every = read_lines(rotated) + read_lines(path)
    ids = [r["i"] for r in every]
    assert ids == list(range(ids[0], 6))  # consecutive, ending with the last record
    assert read_lines(path)[-1]["i"] == 5
    assert path.stat().st_size <= 60


def test_concurrent_appends_do_not_interleave(tmp_path):
    path = tmp_path / "shadow.jsonl"
    log = ShadowLog(path, max_bytes=10_000_000)
    threads = [threading.Thread(target=lambda n=n: [log.append({"n": n, "k": k, "pad": "y" * 500})
                                                    for k in range(50)]) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(read_lines(path)) == 400


def test_non_json_values_are_stringified(tmp_path):
    path = tmp_path / "shadow.jsonl"
    ShadowLog(path, max_bytes=10_000).append({"p": tmp_path})
    assert read_lines(path) == [{"p": str(tmp_path)}]


def test_processes_share_one_rotation_without_losing_records(tmp_path):
    # 4 processes x 200 records of about 135 bytes is about 108 KB: exactly one rotation at 70 KB. Without the
    # inter-process lock, two processes can both rotate and the second rename replaces the fresh .1.
    path = tmp_path / "shadow.jsonl"
    procs = [subprocess.Popen([sys.executable, "-c", WORKER, str(MODULE), str(path), "70000", str(p)])
             for p in range(4)]
    assert [p.wait(timeout=60) for p in procs] == [0, 0, 0, 0]
    lines = read_lines(tmp_path / "shadow.jsonl.1") + read_lines(path)
    assert len(lines) == 800
    assert len({(r["p"], r["k"]) for r in lines}) == 800
