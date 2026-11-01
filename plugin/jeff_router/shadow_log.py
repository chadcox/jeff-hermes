"""Append-only JSONL with one rotated predecessor, shared safely by several Hermes processes.

Each line is one O_APPEND write. Rotation is a check-then-rename, so rotate and append run under an exclusive
flock on <name>.lock (Linux): two processes can never both rotate and overwrite the fresh .1."""
from __future__ import annotations

import fcntl
import json
import os
import threading
from pathlib import Path
from typing import Any


class ShadowLog:
    def __init__(self, path: Path, max_bytes: int) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        with self._lock:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            lock_fd = os.open(self.path.with_name(self.path.name + ".lock"), os.O_WRONLY | os.O_CREAT, 0o600)
            try:
                os.fchmod(lock_fd, 0o600)  # a pre-existing file keeps its mode on open; transcripts are private
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                try:
                    if self.path.stat().st_size + len(line) > self.max_bytes:
                        os.replace(self.path, self.path.with_name(self.path.name + ".1"))
                except FileNotFoundError:
                    pass
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                try:
                    os.fchmod(fd, 0o600)
                    os.write(fd, line)
                finally:
                    os.close(fd)
            finally:
                os.close(lock_fd)  # closing the descriptor releases the flock
