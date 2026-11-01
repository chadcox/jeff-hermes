#!/usr/bin/env python3
"""Summarize jeff_router shadow logs: how often Jeff agreed with Strata's actual next action, by threshold.

Strata's action is a reference, not ground truth, so the report speaks of agreement, never accuracy. Jeff's
answer_directly and ask_user both count as agreeing with any Strata text reply.

Usage: python tools/shadow_report.py [path ...]
With no paths it finds jeff_router's data dir, Hermes' namespaced plugin data dir
$HERMES_HOME/plugin-data/agent-plugin-jeff_router-<digest>/ (HERMES_HOME defaults to ~/.hermes), and reads its
shadow.jsonl.1 and shadow.jsonl without taking the writers' lock (so it never blocks Hermes); if shadow.jsonl.1 is
replaced while reading, it reads both again."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

_OPTIONS = Path(__file__).resolve().parents[1] / "plugin" / "jeff_router" / "options.py"
_spec = importlib.util.spec_from_file_location("jeff_router_options", _OPTIONS)
_options = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _options  # options.py defines dataclasses, which look their module up in sys.modules
_spec.loader.exec_module(_options)
READ_ONLY_TOOLS = _options.READ_ONLY_TOOLS
THRESHOLDS = [round(0.3 + 0.1 * i, 1) for i in range(7)]
DATA_DIR_GLOB = "*jeff_router*"


def _plugin_data() -> Path:
    home = os.environ.get("HERMES_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".hermes") / "plugin-data"


def default_logs() -> list[Path]:
    """shadow.jsonl in every jeff_router data dir; Hermes names the dir agent-plugin-jeff_router-<digest>."""
    return sorted(d / "shadow.jsonl" for d in _plugin_data().glob(DATA_DIR_GLOB) if d.is_dir())


def _parse(data: bytes) -> list[dict]:
    records = []
    for line in data.decode("utf-8", "replace").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def load(paths: list[str]) -> list[dict]:
    return [record for path in paths for record in _parse(Path(path).read_bytes())]


def _inode(path: Path) -> int | None:
    try:
        return path.stat().st_ino
    except FileNotFoundError:
        return None


def load_default(log: Path) -> list[dict]:
    """Read <log>.1 then <log>. A rotation between the two reads replaces <log>.1 (new inode); then read again."""
    rotated = log.with_name(log.name + ".1")
    for _ in range(5):
        before = _inode(rotated)
        blobs = [_read(rotated), _read(log)]
        if _inode(rotated) == before:
            return [record for blob in blobs for record in _parse(blob)]
    raise RuntimeError(f"{rotated} kept changing while it was read; run the report again")


def _read(path: Path) -> bytes:
    """A rotation can remove the current file between listing and reading; that read then counts as empty and the
    inode check above retries."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "—"


def _percentile(values: list[float], q: float) -> str:
    if not values:
        return "—"
    ordered = sorted(values)
    return f"{ordered[max(0, math.ceil(q * len(ordered)) - 1)]:.0f}"


def _picked_name(record: dict) -> str | None:
    tool = record["tool"]
    return tool["names"].get(tool["pick"]) if tool["pick"] not in ("answer_directly", "ask_user") else None


def _p(block: dict) -> float:
    """A block's pick probability; a missing one is never at or above any threshold."""
    p = block.get("p")
    return p if isinstance(p, (int, float)) else -1.0


def _agreement(covered: list[dict]) -> str:
    comparable = [r for r in covered if r.get("tool_match") is not None]
    return f"{_pct(sum(1 for r in comparable if r['tool_match']), len(comparable))} ({len(comparable)})"


def _is_candidate(pick: object) -> bool:
    return isinstance(pick, str) and pick[:1] == "r" and pick[1:].isdigit()


def summarize(records: list[dict]) -> str:
    if not records:
        return "No records."
    lines = ["## Counts", "", "| | count | share |", "|---|---:|---:|", f"| records | {len(records)} |"]
    outcomes = Counter(r.get("outcome") for r in records)
    lines += [f"| {o} | {n} | {_pct(n, len(records))} |" for o, n in outcomes.most_common()]
    ms = [r["jeff_ms"] for r in records
          if r.get("outcome") == "ok" and isinstance(r.get("jeff_ms"), (int, float))]
    ok = [r for r in records if r.get("outcome") == "ok" and r.get("tool")]
    kinds = Counter((r.get("strata") or {}).get("kind") for r in records if r.get("strata"))
    lines += ["", f"Jeff latency (ok): p50 {_percentile(ms, 0.5)} ms, p95 {_percentile(ms, 0.95)} ms. "
              f"Strata actions: tool {kinds.get('tool', 0)}, text {kinds.get('text', 0)}.", ""]

    lines += ["## Tool choice: agreement with Strata", "",
              f"Coverage is the share of all {len(ok)} ok records with tool.p ≥ threshold. Agreement is tool_match "
              "among the covered records that could be compared with Strata (count in brackets).", "",
              "| threshold | coverage | agreement (comparable) | read-only coverage | read-only agreement (comparable) |",
              "|---:|---:|---:|---:|---:|"]
    for t in THRESHOLDS:
        covered = [r for r in ok if _p(r["tool"]) >= t]
        read_only = [r for r in covered if _picked_name(r) in READ_ONLY_TOOLS]
        lines.append(f"| {t} | {_pct(len(covered), len(ok))} ({len(covered)}) | {_agreement(covered)} | "
                     f"{_pct(len(read_only), len(ok))} ({len(read_only)}) | {_agreement(read_only)} |")

    full = [r for r in ok if _picked_name(r) == "read_file" and isinstance(r.get("read_target"), dict)
            and _is_candidate(r["read_target"].get("pick"))]
    lines += ["", "## Full read_file call and calls that could be skipped", "",
              "Records where Jeff picked read_file with tool.p ≥ t (rows) and a read_target candidate rK with "
              "read_target.p ≥ t2 (columns). Each cell is answered / match / mismatch: answered is how many calls Jeff "
              "would have answered locally, match and mismatch compare its call with Strata's first call (a text "
              "answer or another tool is a mismatch), and the rest of answered are undecided (Strata called "
              "read_file with a path that could not be resolved, or its action could not be read).", "",
              "| t \\ t2 | " + " | ".join(str(t2) for t2 in THRESHOLDS) + " |",
              "|---:|" + "---:|" * len(THRESHOLDS)]
    for t in THRESHOLDS:
        cells = []
        for t2 in THRESHOLDS:
            covered = [r for r in full if _p(r["tool"]) >= t and _p(r["read_target"]) >= t2]
            matched = sum(1 for r in covered if r.get("read_match") is True)
            mismatched = sum(1 for r in covered if r.get("read_match") is False
                             or (r.get("read_match") is None and r.get("tool_match") is False))
            cells.append(f"{len(covered)} / {matched} / {mismatched}")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", f"Strata actions in the log: {sum(kinds.values())}. A match is a call Jeff could have answered "
              "the way Strata did. Mismatches are not necessarily Jeff errors: review a sample of high-confidence "
              "mismatches by hand before trusting any threshold."]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    log = None
    if not argv:
        logs = default_logs()
        if len(logs) > 1:
            print("Several jeff_router data dirs; pass the shadow.jsonl to report on:", file=sys.stderr)
            print("\n".join(f"  {path}" for path in logs), file=sys.stderr)
            return 1
        if not logs:
            print(f"No records in {_plugin_data() / DATA_DIR_GLOB / 'shadow.jsonl'}(.1).")
            return 1
        log = logs[0]
    try:
        records = load(argv) if argv else load_default(log)
    except (OSError, RuntimeError) as error:
        print(f"Cannot read the shadow log: {error}", file=sys.stderr)
        return 1
    if not records:
        print("No records found." if argv else f"No records in {log}(.1).")
        return 1
    print(summarize(records))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
