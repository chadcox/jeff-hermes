# jeff_router v1 (shadow mode) Implementation Plan

> **Plan drift (2026-10-31).** The code on branch `jeff-router-v1` differs from the steps below; the code and the
> spec are authoritative. Where they differ:
> - `log_max_mb` is an int in `plugin.yaml` (`type: int`), not a float.
> - `tests/conftest.py` serves fake Jeff from a `_TestServer(ThreadingHTTPServer)` subclass with `daemon_threads` and
>   `block_on_close` set on the subclass, instead of patching `ThreadingHTTPServer` class attributes.
> - `state.fit_state` is linear: each conversation entry is serialized once, and `user_message` shrinks by
>   `ceil(excess / 6)` per step (a character serializes to at most 6), not `room -= excess`.
> - `options.py` caps filesystem probes per question at `MAX_PATH_PROBES = 64`, probes each raw path once, checks only
>   the newest `MAX_ALREADY_READ = 200` `read_file` calls for re-reads, and its path pattern accepts single-segment
>   `/x`, `~/x`, `./x` and `../x`.
> - `jeff_client.py` is hardened: no proxies (`ProxyHandler({})`), `http.client.HTTPException` and request encoding
>   inside the `try`, `jeff_ms` kept on failures.
> - `compare.py` returns `None` (undecidable) instead of `False` when a pick is unknown, Strata's tool has no name, or
>   Strata's `read_file` path cannot be resolved; logged argument strings are capped at 2,000 characters.
> - `register()` registers nothing (one warning) when the plugin context lacks `get_config`, `register_middleware`
>   or `state.data_dir`.
> - `deploy/jeff-serve.service` warms up with two `ExecStartPost` lines (wait for `/health` ready, then one example
>   request) and `TimeoutStartSec=180`; the probe also warms jeff-serve first.
> - The report shows tool agreement among comparable records with coverage share and counts, and a t × t2 grid of
>   answered / match / mismatch counts for full `read_file` calls (not one shared threshold). A text answer or another
>   tool from Strata counts as a mismatch; undecided records are counted as answered only.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Hermes plugin that, on every main-turn model call, asks Jeff (on the GPU) what the agent will do next, lets the call go to Strata unchanged, and logs Jeff's pick beside Strata's actual action. A report script turns the log into accuracy-by-threshold tables.

**Architecture:**
- An always-on `jeff-serve` systemd user unit serves Jeff on `127.0.0.1:8765`.
- The plugin `jeff_router` registers a synchronous `llm_execution` middleware. It builds Jeff's request on the agent thread and sends it from a bounded two-slot pool while `next_call()` runs. It returns Strata's response at once; a done-callback appends one JSONL record when Jeff finishes.
- The plugin uses only the standard library and never alters the request or the response.

**Tech Stack:**
- Python 3.14: the Hermes runtime at `~/.hermes/tools/python-3.14.9+20261021-linux-x64/bin/python3`.
- Standard library only in the plugin.
- pytest ≥ 8, run through `uvx`, so nothing is installed into Hermes.
- systemd user units.

**Spec:** `docs/design/2026-10-30-shadow-design.md`. Executors read it alongside this plan.

## Global Constraints

- Never modify `~/.hermes/hermes-agent`. `git -C ~/.hermes/hermes-agent status --short` must print nothing at the end.
- Never use `~/.hermes/hermes-agent/venv/bin/python`; it is stale (3.11, missing `ruamel`). Hermes probes run with `~/.hermes/tools/python-3.14.9+20261021-linux-x64/bin/python3 -I`, with the repo on `sys.path` and `hermes_bootstrap` imported.
- The plugin imports only the standard library and its own modules. No Hermes imports.
- The plugin directory name is `jeff_router`, which must sort before `warm_compaction`. Do **not** add `requires_plugins`.
- Call `next_call()` with **no argument**, so the request object passes on untouched (warm_compaction's terminal refuses rewritten requests).
- `ctx.session_cwd()` reads task-local context, so call it on the agent thread, never in the pool thread.
- The middleware can never raise its own exception into Hermes. An exception raised by `next_call()` propagates unchanged and nothing is logged for that call.
- Jeff runs on the GPU: `JEFF_DEVICE=cuda`, `.venv-gpu`, base `models/jeff-base-v1.3`, adapters `models/adapters`, port `8765`, `JEFF_QUEUE_MS=250`.
- Log path: `<ctx.state.data_dir>/shadow.jsonl`, resolved once in `register()`.
- Settings and defaults:
  - `enabled=true`
  - `url="http://127.0.0.1:8765"`
  - `timeout_s=2.0`
  - `max_request_chars=24000`
  - `log_max_mb=50`
- Ask the user before restarting `hermes-gateway.service` or any running Hermes app.

Test command, used throughout (run from the repo root):

```bash
PYTEST="uvx -q --python $HOME/.hermes/tools/python-3.14.9+20261021-linux-x64/bin/python3 --from pytest>=8 pytest"
```

---

## File structure

| Path | Responsibility |
|---|---|
| `pytest.ini` | Test discovery: `testpaths = tests` |
| `tests/conftest.py` | Loads `plugin/jeff_router` as package `jeff_router_under_test`; fake jeff-serve fixture |
| `plugin/jeff_router/plugin.yaml` | Manifest |
| `plugin/jeff_router/__init__.py` | `register(ctx)` |
| `plugin/jeff_router/settings.py` | `DEFAULTS`, `read_settings(get_config)` |
| `plugin/jeff_router/jeff_client.py` | `JeffResult`, `JeffClient.ask(body)`: HTTP plus outcome mapping |
| `plugin/jeff_router/options.py` | Read-only set, instruction texts, `build_tool_options`, `build_read_options`, `normalize_path` |
| `plugin/jeff_router/state.py` | `build_state(messages)`, `fit_state(state, budget)` |
| `plugin/jeff_router/compare.py` | `strata_action(response)`, `pick(answer)`, `tool_match`, `read_match` |
| `plugin/jeff_router/shadow_log.py` | `ShadowLog.append(record)` with rotation |
| `plugin/jeff_router/middleware.py` | `ShadowRouter`: gating, flow, record building |
| `tools/shadow_report.py` | Markdown report from `shadow.jsonl*` |
| `deploy/jeff-serve.service` | systemd user unit |
| `probes/shadow_probe.py` | End-to-end check with a real `AIAgent` and the real jeff-serve |
| `README.md` | How to run the service, the tests, the report, and how to enable or disable the plugin |

---

### Task 1: Test harness, manifest, settings, `register()` skeleton

**Files:**
- Create: `pytest.ini`, `tests/conftest.py`, `plugin/jeff_router/plugin.yaml`, `plugin/jeff_router/__init__.py`, `plugin/jeff_router/settings.py`
- Test: `tests/test_settings.py`

**Interfaces:**
- Produces:
  - `settings.DEFAULTS: dict[str, object]`
  - `settings.read_settings(get_config: Callable[[str, object], object]) -> dict[str, object]`
  - conftest: package `jeff_router_under_test` importable by tests, and fixture `fake_jeff` returning a `FakeJeff` with `.url`, `.requests: list[dict]`, `.reply`, `.delay`

- [ ] **Step 1: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -q
```

- [ ] **Step 2: Create `tests/conftest.py`**

```python
"""Load plugin/jeff_router as a package the way Hermes does (spec_from_file_location with a search path),
and provide a real local HTTP server standing in for jeff-serve."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO / "plugin" / "jeff_router"
PACKAGE = "jeff_router_under_test"


def _load_plugin():
    if PACKAGE in sys.modules:
        return sys.modules[PACKAGE]
    spec = importlib.util.spec_from_file_location(
        PACKAGE, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
    return module


_load_plugin()


def ok_body(answers: dict[str, dict[str, float]], input_tokens: int = 100) -> dict:
    """A jeff-serve success body: answers.<q>.probabilities plus choice, like the real server."""
    return {"answers": {q: {"probabilities": probs, "choice": max(probs, key=probs.get), "confidence": 0.5}
                        for q, probs in answers.items()},
            "usage": {"input_tokens": input_tokens, "output_tokens": 0, "orders": 1}}


class FakeJeff:
    def __init__(self) -> None:
        self.url = ""
        self.requests: list[dict] = []
        self.delay = 0.0
        # (status, body) or a callable taking the request body and returning (status, body).
        # body may be a dict (sent as JSON) or bytes (sent as is).
        self.reply = (200, ok_body({"tool": {"answer_directly": 1.0}}))


@pytest.fixture
def fake_jeff():
    state = FakeJeff()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (http.server naming)
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            state.requests.append(body)
            if state.delay:
                time.sleep(state.delay)
            status, payload = state.reply(body) if callable(state.reply) else state.reply
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer.block_on_close = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()
```

- [ ] **Step 3: Write the failing test `tests/test_settings.py`**

```python
from jeff_router_under_test.settings import DEFAULTS, read_settings


def test_defaults_when_nothing_configured():
    assert read_settings(lambda key, default: default) == DEFAULTS


def test_valid_values_are_used():
    given = {"enabled": False, "url": "http://127.0.0.1:9999", "timeout_s": 1.5, "max_request_chars": 12000,
             "log_max_mb": 5}
    assert read_settings(lambda key, default: given.get(key, default)) == given


def test_invalid_values_fall_back_to_defaults():
    given = {"enabled": "yes", "url": "", "timeout_s": -1, "max_request_chars": 0, "log_max_mb": None}
    assert read_settings(lambda key, default: given.get(key, default)) == DEFAULTS


def test_bool_is_not_accepted_as_a_number():
    assert read_settings(lambda key, default: True if key == "timeout_s" else default)["timeout_s"] == 2.0


def test_get_config_errors_fall_back_to_defaults():
    def broken(key, default):
        raise RuntimeError("config unreadable")
    assert read_settings(broken) == DEFAULTS
```

- [ ] **Step 4: Run it; expect a collection failure**

Run: `$PYTEST tests/test_settings.py`
Expected: ERROR in conftest loading (`plugin/jeff_router/__init__.py` missing).

- [ ] **Step 5: Create the manifest `plugin/jeff_router/plugin.yaml`**

```yaml
manifest_version: 2
name: jeff_router
version: 0.1.0
description: "Shadow mode: asks Jeff (jeff-serve) what the agent will do next on each main-turn model call and logs Jeff's pick beside Strata's actual action. Never changes requests or responses."
kind: standalone
author: Chad Cox
tags: [jeff, routing, shadow]
provides_middleware:
  - llm_execution
config_schema:
  enabled:
    type: bool
    default: true
    description: Turn the shadow router off without removing the plugin.
  url:
    type: str
    default: "http://127.0.0.1:8765"
    description: Base URL of jeff-serve.
  timeout_s:
    type: float
    default: 2.0
    description: Client timeout in seconds for one Jeff request. Off the critical path in shadow mode.
  max_request_chars:
    type: int
    default: 24000
    description: Character budget for the whole Jeff request, so it stays under jeff-serve's 8192-token limit.
  log_max_mb:
    type: float
    default: 50
    description: Rotate shadow.jsonl to shadow.jsonl.1 above this size in megabytes.
```

- [ ] **Step 6: Create `plugin/jeff_router/settings.py`**

```python
"""jeff_router settings: read once in register(); invalid values fall back to the defaults."""
from __future__ import annotations

from typing import Any, Callable

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "url": "http://127.0.0.1:8765",
    "timeout_s": 2.0,
    "max_request_chars": 24000,
    "log_max_mb": 50,
}


def _valid(default: Any, value: Any) -> bool:
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, (int, float)):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
    if isinstance(default, str):
        return isinstance(value, str) and bool(value.strip())
    return False


def read_settings(get_config: Callable[[str, Any], Any]) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for key, default in DEFAULTS.items():
        try:
            value = get_config(key, default)
        except Exception:
            value = default
        settings[key] = value if _valid(default, value) else default
    return settings
```

- [ ] **Step 7: Create `plugin/jeff_router/__init__.py`** (a skeleton; Task 7 completes `register`)

```python
"""jeff_router: shadow-mode Jeff router for Hermes Agent (see docs/superpowers/specs in the jeff-hermes repo)."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register(ctx: Any) -> None:
    """Hermes entry point. Completed in Task 7."""
    from .settings import read_settings

    settings = read_settings(ctx.get_config)
    if not settings["enabled"]:
        logger.info("jeff_router: disabled in settings; registering nothing")
        return
```

- [ ] **Step 8: Run the tests; expect them to pass**

Run: `$PYTEST tests/test_settings.py`
Expected: `5 passed`.

- [ ] **Step 9: Commit**

```bash
git add pytest.ini tests/conftest.py tests/test_settings.py plugin/jeff_router
git commit -m "jeff_router: test harness, manifest, settings"
```

---

### Task 2: Jeff HTTP client with outcome mapping

**Files:**
- Create: `plugin/jeff_router/jeff_client.py`
- Test: `tests/test_jeff_client.py`

**Interfaces:**
- Produces:
  - `JeffResult(outcome: str, answers: dict | None = None, input_tokens: int | None = None, ms: float | None = None, detail: str = "")`, frozen dataclass.
  - `JeffClient(base_url: str, timeout_s: float)` with `.ask(body: dict) -> JeffResult`. It never raises.
  - Outcomes: `ok`, `timeout`, `refused`, `busy`, `too_long`, `bad_request`, `server_error`, `bad_response`.

- [ ] **Step 1: Write the failing tests `tests/test_jeff_client.py`**

```python
import socket

from conftest import ok_body
from jeff_router_under_test.jeff_client import JeffClient

BODY = {"model": "tools", "state": {"user_message": "hi"},
        "questions": {"tool": {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}}}}


def test_ok_returns_answers_tokens_and_latency(fake_jeff):
    fake_jeff.reply = (200, ok_body({"tool": {"a": 0.8, "b": 0.2}}, input_tokens=321))
    result = JeffClient(fake_jeff.url, 2.0).ask(BODY)
    assert result.outcome == "ok"
    assert result.answers["tool"]["probabilities"] == {"a": 0.8, "b": 0.2}
    assert result.input_tokens == 321
    assert result.ms is not None and result.ms >= 0
    assert fake_jeff.requests == [BODY]


def test_trailing_slash_in_base_url(fake_jeff):
    assert JeffClient(fake_jeff.url + "/", 2.0).ask(BODY).outcome == "ok"


def test_busy_529(fake_jeff):
    fake_jeff.reply = (529, {"detail": "The model is busy. Retry shortly."})
    assert JeffClient(fake_jeff.url, 2.0).ask(BODY).outcome == "busy"


def test_422_token_limit_is_too_long(fake_jeff):
    fake_jeff.reply = (422, {"detail": "Question branch exceeds the 8192-token limit; no input was truncated."})
    assert JeffClient(fake_jeff.url, 2.0).ask(BODY).outcome == "too_long"


def test_other_422_is_bad_request(fake_jeff):
    fake_jeff.reply = (422, {"detail": [{"loc": ["body", "model"], "msg": "Field required", "type": "missing"}]})
    assert JeffClient(fake_jeff.url, 2.0).ask(BODY).outcome == "bad_request"


def test_503_is_server_error(fake_jeff):
    fake_jeff.reply = (503, {"detail": "The model is not ready."})
    assert JeffClient(fake_jeff.url, 2.0).ask(BODY).outcome == "server_error"


def test_timeout(fake_jeff):
    fake_jeff.delay = 0.5
    result = JeffClient(fake_jeff.url, 0.1).ask(BODY)
    assert result.outcome == "timeout"


def test_connection_refused():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert JeffClient(f"http://127.0.0.1:{port}", 1.0).ask(BODY).outcome == "refused"


def test_non_json_body_is_bad_response(fake_jeff):
    fake_jeff.reply = (200, b"not json")
    assert JeffClient(fake_jeff.url, 2.0).ask(BODY).outcome == "bad_response"


def test_missing_probabilities_is_bad_response(fake_jeff):
    fake_jeff.reply = (200, {"answers": {"tool": {"choice": "a"}}, "usage": {"input_tokens": 3}})
    assert JeffClient(fake_jeff.url, 2.0).ask(BODY).outcome == "bad_response"
```

- [ ] **Step 2: Run them; expect failure**

Run: `$PYTEST tests/test_jeff_client.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'jeff_router_under_test.jeff_client'`.

- [ ] **Step 3: Implement `plugin/jeff_router/jeff_client.py`**

```python
"""Minimal jeff-serve client. Every failure becomes an outcome string; ask() never raises."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JeffResult:
    outcome: str
    answers: dict[str, Any] | None = None
    input_tokens: int | None = None
    ms: float | None = None
    detail: str = ""


def _http_outcome(code: int, body: str) -> str:
    if code == 529:
        return "busy"
    if code == 422 and "token limit" in body:
        return "too_long"
    if 400 <= code < 500:
        return "bad_request"
    return "server_error"


def _valid_answers(payload: Any) -> bool:
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict) or not answers:
        return False
    for answer in answers.values():
        probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
        if not isinstance(probabilities, dict) or not probabilities:
            return False
        if not all(isinstance(p, (int, float)) and not isinstance(p, bool) for p in probabilities.values()):
            return False
    return True


class JeffClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.timeout_s = timeout_s

    def ask(self, body: dict[str, Any]) -> JeffResult:
        request = urllib.request.Request(self.url, json.dumps(body).encode("utf-8"),
                                         {"Content-Type": "application/json"}, method="POST")
        started = time.perf_counter()

        def elapsed() -> float:
            return round((time.perf_counter() - started) * 1000, 1)

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            try:
                text = error.read().decode("utf-8", "replace")
            except Exception:
                text = ""
            return JeffResult(_http_outcome(error.code, text), ms=elapsed(), detail=f"HTTP {error.code}")
        except urllib.error.URLError as error:
            outcome = "timeout" if isinstance(error.reason, TimeoutError) else "refused"
            return JeffResult(outcome, ms=elapsed(), detail=type(error.reason).__name__)
        except TimeoutError:
            return JeffResult("timeout", ms=elapsed(), detail="read timeout")
        except OSError as error:
            return JeffResult("refused", ms=elapsed(), detail=type(error).__name__)
        ms = elapsed()
        try:
            payload = json.loads(raw)
        except ValueError:
            return JeffResult("bad_response", ms=ms, detail="non-JSON body")
        if not _valid_answers(payload):
            return JeffResult("bad_response", ms=ms, detail="missing answers/probabilities")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        tokens = usage.get("input_tokens")
        return JeffResult("ok", answers=payload["answers"],
                          input_tokens=tokens if isinstance(tokens, int) else None, ms=ms)
```

- [ ] **Step 4: Run them; expect them to pass**

Run: `$PYTEST tests/test_jeff_client.py`
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add plugin/jeff_router/jeff_client.py tests/test_jeff_client.py
git commit -m "jeff_router: jeff-serve client with outcome mapping"
```

---

### Task 3: Option builders (`tool` and `read_target`)

**Files:**
- Create: `plugin/jeff_router/options.py`
- Test: `tests/test_options.py`

**Interfaces:**
- Produces:
  - Constants: `READ_ONLY_TOOLS: frozenset[str]`, `TOOL_INSTRUCTIONS: str`, `READ_INSTRUCTIONS: str`, `MAX_TOOLS = 150` (the adapter card's training range, on top of `answer_directly` and `ask_user`; the adapter allows 252 options), `MAX_READ_CANDIDATES = 10`.
  - `ToolOptions(criteria: dict[str, str], names: dict[str, str], truncated: bool)`. `names` maps `tK` to the tool name.
  - `build_tool_options(tools: list) -> ToolOptions`
  - `ReadOptions(criteria: dict[str, str], paths: dict[str, str])`. `paths` maps `rK` to an absolute resolved path.
  - `build_read_options(messages: list, tool_names: set[str], cwd: str) -> ReadOptions | None`
  - `normalize_path(raw: object, cwd: str) -> str | None`: absolute and resolved, without requiring the file to exist.

- [ ] **Step 1: Write the failing tests `tests/test_options.py`**

```python
import json

from jeff_router_under_test.options import (
    MAX_TOOLS, READ_INSTRUCTIONS, READ_ONLY_TOOLS, TOOL_INSTRUCTIONS, build_read_options, build_tool_options,
    normalize_path,
)


def tool(name, description="Does a thing. More text.", required=("path",)):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "required": list(required)}}}


def test_tool_options_order_and_format():
    opts = build_tool_options([tool("read_file", "Read a text file from disk. Supports paging."),
                               tool("terminal", "Run a shell command.", required=("command",))])
    assert list(opts.criteria) == ["answer_directly", "ask_user", "t1", "t2"]
    assert opts.criteria["answer_directly"] == "No tool is needed: answer the user directly from the conversation"
    assert opts.criteria["ask_user"] == "A tool is needed but required information is missing: ask the user first"
    assert opts.criteria["t1"] == "read_file(path): Read a text file from disk."
    assert opts.criteria["t2"] == "terminal(command): Run a shell command."
    assert opts.names == {"t1": "read_file", "t2": "terminal"}
    assert opts.truncated is False


def test_long_description_is_cut_to_120_chars():
    opts = build_tool_options([tool("x", "word " * 100, required=())])
    description = opts.criteria["t1"].split(": ", 1)[1]
    assert len(description) == 120 and description.endswith("…")


def test_malformed_and_duplicate_tools_are_skipped():
    opts = build_tool_options([{"type": "function"}, tool("a"), tool("a"), "junk", tool("b")])
    assert opts.names == {"t1": "a", "t2": "b"}


def test_over_150_tools_keeps_read_only_first_in_request_order():
    tools = [tool(f"w{i}") for i in range(160)] + [tool("read_file"), tool("search_files")]
    opts = build_tool_options(tools)
    names = list(opts.names.values())
    assert opts.truncated is True
    assert len(names) == MAX_TOOLS
    assert "read_file" in names and "search_files" in names
    assert names[:3] == ["w0", "w1", "w2"]  # request order is preserved
    assert "w148" not in names  # the last write tools make room for the two read-only ones


def test_instruction_texts_are_the_adapter_wording():
    assert TOOL_INSTRUCTIONS.startswith("Which tool should the agent call next to handle the user's latest message?")
    assert "none of these" in READ_INSTRUCTIONS
    assert "read_file" in READ_ONLY_TOOLS and "terminal" not in READ_ONLY_TOOLS


def msg(role, content, **extra):
    return {"role": role, "content": content, **extra}


def test_read_candidates_from_user_message(tmp_path):
    target = tmp_path / "notes.md"
    target.write_text("hello")
    messages = [msg("system", f"ignore {tmp_path}/system.md"), msg("user", f"What's in {target}?")]
    opts = build_read_options(messages, {"read_file"}, cwd="")
    assert opts.paths == {"r1": str(target.resolve())}
    assert list(opts.criteria) == ["r1", "none_of_these"]
    assert opts.criteria["r1"] == f'read_file(path="{target.resolve()}"): read notes.md'


def test_relative_paths_resolve_against_cwd_and_missing_cwd_skips_them(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("x")
    messages = [msg("user", "open docs/a.md please")]
    assert build_read_options(messages, {"read_file"}, cwd=str(tmp_path)).paths == {
        "r1": str((tmp_path / "docs" / "a.md").resolve())}
    assert build_read_options(messages, {"read_file"}, cwd="") is None


def test_already_read_without_paging_is_excluded_but_paged_read_is_not(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("a")
    b.write_text("b")
    call = lambda path, **kw: {"id": "c", "type": "function",  # noqa: E731
                               "function": {"name": "read_file", "arguments": json.dumps({"path": path, **kw})}}
    messages = [msg("user", f"compare {a} and {b}"),
                msg("assistant", None, tool_calls=[call(str(a)), call(str(b), offset=10)]),
                msg("tool", "1|a", tool_call_id="c")]
    opts = build_read_options(messages, {"read_file"}, cwd="")
    assert list(opts.paths.values()) == [str(b.resolve())]


def test_directories_missing_files_urls_and_big_files_are_rejected(tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"0" * (2 * 1024 * 1024 + 1))
    messages = [msg("user", f"{tmp_path} {tmp_path}/missing.txt https://example.com/a/b.txt {big}")]
    assert build_read_options(messages, {"read_file"}, cwd="") is None


def test_no_read_file_tool_means_no_question(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")
    assert build_read_options([msg("user", str(f))], {"terminal"}, cwd="") is None


def test_candidates_are_capped_at_ten_newest_first(tmp_path):
    files = []
    for i in range(12):
        f = tmp_path / f"f{i}.txt"
        f.write_text("x")
        files.append(f)
    messages = [msg("assistant", " ".join(str(f) for f in files[:6])),
                msg("user", " ".join(str(f) for f in files[6:]))]
    opts = build_read_options(messages, {"read_file"}, cwd="")
    assert len(opts.paths) == 10
    assert opts.paths["r1"] == str(files[6].resolve())  # the latest user message comes first


def test_trailing_punctuation_is_stripped(tmp_path):
    f = tmp_path / "x.md"
    f.write_text("x")
    assert build_read_options([msg("user", f"see {f}.")], {"read_file"}, cwd="").paths == {"r1": str(f.resolve())}


def test_normalize_path_accepts_missing_files(tmp_path):
    assert normalize_path("nope/x.txt", str(tmp_path)) == str((tmp_path / "nope" / "x.txt").resolve())
    assert normalize_path(None, str(tmp_path)) is None
    assert normalize_path("rel.txt", "") is None
```

- [ ] **Step 2: Run them; expect failure**

Run: `$PYTEST tests/test_options.py`
Expected: FAIL with `ModuleNotFoundError ... options`.

- [ ] **Step 3: Implement `plugin/jeff_router/options.py`**

```python
"""Questions for Jeff: the tool list (tools adapter format) and code-built read_file candidates."""
from __future__ import annotations

import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Copy of Hermes' NO_EFFECT_TOOL_NAMES (agent/tool_result_classification.py:16-20); not imported on purpose.
READ_ONLY_TOOLS = frozenset({
    "read_file", "search_files", "session_search", "skill_view", "skills_list",
    "web_extract", "web_search", "vision_analyze", "browser_snapshot",
    "browser_get_images", "browser_console", "read_terminal",
})
TOOL_INSTRUCTIONS = (
    "Which tool should the agent call next to handle the user's latest message? Use the conversation for context. "
    "If no tool is needed, choose answer directly. If a tool is needed but information it requires is missing, "
    "choose ask the user."
)
ANSWER_DIRECTLY = "No tool is needed: answer the user directly from the conversation"
ASK_USER = "A tool is needed but required information is missing: ask the user first"
READ_INSTRUCTIONS = (
    "Which file should the agent read next to handle the user's latest message? If none of these files is the "
    "right one, choose none of these."
)
NONE_OF_THESE = "None of these: the agent needs a different file or a different tool"
MAX_TOOLS = 150
MAX_READ_CANDIDATES = 10
MAX_READ_BYTES = 2 * 1024 * 1024
DESCRIPTION_CHARS = 120
SCAN_MESSAGES = 12
_PATH = re.compile(r"(?<![\w:/.~-])(?:~|\.{1,2})?/?[A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)+")


@dataclass(frozen=True)
class ToolOptions:
    criteria: dict[str, str]
    names: dict[str, str]
    truncated: bool


@dataclass(frozen=True)
class ReadOptions:
    criteria: dict[str, str]
    paths: dict[str, str]


def _first_sentence(text: str) -> str:
    text = " ".join(str(text).split())
    match = re.search(r"(?<=[.!?])\s", text)
    sentence = text[:match.start()] if match else text
    if len(sentence) > DESCRIPTION_CHARS:
        sentence = sentence[:DESCRIPTION_CHARS - 1].rstrip() + "…"
    return sentence


def _signature(tool: Any) -> tuple[str, str] | None:
    function = tool.get("function") if isinstance(tool, dict) else None
    name = function.get("name") if isinstance(function, dict) else None
    if not isinstance(name, str) or not name:
        return None
    parameters = function.get("parameters")
    required = parameters.get("required") if isinstance(parameters, dict) else None
    args = ", ".join(str(r) for r in required) if isinstance(required, list) else ""
    description = _first_sentence(function.get("description") or "")
    return name, f"{name}({args}): {description}" if description else f"{name}({args})"


def build_tool_options(tools: list[Any]) -> ToolOptions:
    signatures: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tool in tools if isinstance(tools, list) else []:
        signature = _signature(tool)
        if signature and signature[0] not in seen:
            seen.add(signature[0])
            signatures.append(signature)
    truncated = len(signatures) > MAX_TOOLS
    if truncated:
        ranked = ([s for s in signatures if s[0] in READ_ONLY_TOOLS]
                  + [s for s in signatures if s[0] not in READ_ONLY_TOOLS])
        keep = {name for name, _ in ranked[:MAX_TOOLS]}
        signatures = [s for s in signatures if s[0] in keep]
    criteria = {"answer_directly": ANSWER_DIRECTLY, "ask_user": ASK_USER}
    names: dict[str, str] = {}
    for index, (name, text) in enumerate(signatures, 1):
        criteria[f"t{index}"] = text
        names[f"t{index}"] = name
    return ToolOptions(criteria, names, truncated)


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part["text"] for part in content
                         if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


def normalize_path(raw: Any, cwd: str) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        path = Path(raw.strip()).expanduser()
        if not path.is_absolute():
            if not cwd:
                return None
            path = Path(cwd) / path
        return str(path.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return None


def _existing_file(raw: str, cwd: str) -> str | None:
    normalized = normalize_path(raw, cwd)
    if normalized is None:
        return None
    try:
        info = Path(normalized).stat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_READ_BYTES:
        return None
    return normalized


def _already_read(messages: list[Any], cwd: str) -> set[str]:
    read: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or function.get("name") != "read_file":
                continue
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                continue
            if isinstance(args, dict) and "offset" not in args and "limit" not in args:
                normalized = normalize_path(args.get("path"), cwd)
                if normalized:
                    read.add(normalized)
    return read


def _texts_newest_first(messages: list[Any]) -> list[str]:
    convo = [m for m in messages if isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool")]
    last_user = next((m for m in reversed(convo) if m.get("role") == "user"), None)
    texts = [message_text(last_user.get("content"))] if last_user else []
    texts += [message_text(m.get("content")) for m in reversed(convo[-SCAN_MESSAGES:]) if m is not last_user]
    return texts


def build_read_options(messages: list[Any], tool_names: set[str], cwd: str) -> ReadOptions | None:
    if "read_file" not in tool_names or not isinstance(messages, list):
        return None
    done = _already_read(messages, cwd)
    found: list[str] = []
    for text in _texts_newest_first(messages):
        for match in _PATH.finditer(text):
            path = _existing_file(match.group(0).rstrip(".,:;"), cwd)
            if path and path not in done and path not in found:
                found.append(path)
            if len(found) == MAX_READ_CANDIDATES:
                break
        if len(found) == MAX_READ_CANDIDATES:
            break
    if not found:
        return None
    criteria: dict[str, str] = {}
    paths: dict[str, str] = {}
    for index, path in enumerate(found, 1):
        key = f"r{index}"
        criteria[key] = f'read_file(path="{path}"): read {Path(path).name}'
        paths[key] = path
    criteria["none_of_these"] = NONE_OF_THESE
    return ReadOptions(criteria, paths)
```

- [ ] **Step 4: Run them; expect them to pass**

Run: `$PYTEST tests/test_options.py`
Expected: `13 passed`.

- [ ] **Step 5: Commit**

```bash
git add plugin/jeff_router/options.py tests/test_options.py
git commit -m "jeff_router: tool options and read_file candidates"
```

---

### Task 4: State rendering and trimming

**Files:**
- Create: `plugin/jeff_router/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: `options.message_text(content) -> str`
- Produces:
  - `AGENT: str`
  - `build_state(messages: list) -> dict`, with keys `agent`, `conversation: list[{role, text}]`, `user_message`
  - `fit_state(state: dict, budget: int) -> dict | None`. The JSON-serialized state is guaranteed to be at most `budget` characters, or the result is `None`.

- [ ] **Step 1: Write the failing tests `tests/test_state.py`**

```python
import json

from jeff_router_under_test.state import AGENT, build_state, fit_state


def call(name, args):
    return {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_layout_and_roles():
    messages = [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": [{"type": "text", "text": "read the notes"}, {"type": "image_url"}]},
        {"role": "assistant", "content": None, "tool_calls": [call("read_file", {"path": "/x"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "line1\nline2"},
    ]
    state = build_state(messages)
    assert list(state) == ["agent", "conversation", "user_message"]
    assert state["agent"] == AGENT
    assert state["user_message"] == "read the notes"
    assert state["conversation"] == [
        {"role": "user", "text": "earlier question"},
        {"role": "assistant", "text": "earlier answer"},
        {"role": "assistant", "text": '-> read_file({"path": "/x"})'},
        {"role": "tool", "text": "line1\nline2"},
    ]


def test_only_last_six_before_the_user_message():
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
    messages.append({"role": "user", "content": "now"})
    state = build_state(messages)
    assert [e["text"] for e in state["conversation"]] == ["m4", "m5", "m6", "m7", "m8", "m9"]


def test_tool_output_keeps_last_40_lines_of_200_chars():
    output = "\n".join(f"{i}:" + "x" * 300 for i in range(100))
    state = build_state([{"role": "user", "content": "go"}, {"role": "tool", "content": output}])
    lines = state["conversation"][0]["text"].splitlines()
    assert len(lines) == 40 and lines[0].startswith("60:") and all(len(line) <= 200 for line in lines)


def test_tool_call_line_is_cut_to_300_chars():
    messages = [{"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [call("write_file", {"content": "y" * 1000})]}]
    assert len(build_state(messages)["conversation"][0]["text"]) == 300


def test_no_user_message():
    assert build_state([{"role": "system", "content": "s"}])["user_message"] == ""


def size(state):
    return len(json.dumps(state, ensure_ascii=False))


def test_fit_state_returns_same_content_when_it_fits():
    state = {"agent": "a", "conversation": [{"role": "user", "text": "x"}], "user_message": "u"}
    assert fit_state(state, 10_000) == state


def test_fit_state_drops_oldest_conversation_first():
    state = {"agent": "a", "conversation": [{"role": "user", "text": "o" * 100},
                                            {"role": "user", "text": "n" * 100}], "user_message": "u"}
    fitted = fit_state(state, size(state) - 50)
    assert [e["text"][0] for e in fitted["conversation"]] == ["n"]
    assert size(fitted) <= size(state) - 50
    assert len(state["conversation"]) == 2  # input is not mutated


def test_fit_state_cuts_user_message_head_and_tail():
    state = {"agent": "a", "conversation": [], "user_message": "H" * 500 + "T" * 500}
    fitted = fit_state(state, 300)
    assert size(fitted) <= 300
    assert fitted["user_message"].startswith("H") and fitted["user_message"].endswith("T")
    assert "…" in fitted["user_message"]


def test_fit_state_handles_escaped_characters():
    state = {"agent": "a", "conversation": [], "user_message": '"\\' * 400}
    fitted = fit_state(state, 200)
    assert fitted is not None and size(fitted) <= 200


def test_fit_state_gives_up_when_budget_is_tiny():
    assert fit_state({"agent": "a" * 100, "conversation": [], "user_message": "u"}, 20) is None
```

- [ ] **Step 2: Run them; expect failure**

Run: `$PYTEST tests/test_state.py`
Expected: FAIL with `ModuleNotFoundError ... state`.

- [ ] **Step 3: Implement `plugin/jeff_router/state.py`**

```python
"""The tools adapter's state (agent, conversation, user_message) built from Hermes request messages."""
from __future__ import annotations

import json
from typing import Any

from .options import message_text

AGENT = "A personal assistant with shell, file, web, memory and other tools on the user's Linux workstation."
MAX_CONVERSATION = 6
TOOL_LINES = 40
LINE_CHARS = 200
CALL_CHARS = 300
ROLES = ("user", "assistant", "tool")


def _tail(text: str) -> str:
    return "\n".join(line[:LINE_CHARS] for line in text.splitlines()[-TOOL_LINES:])


def _render(message: dict[str, Any]) -> dict[str, str]:
    role = message["role"]
    if role == "tool":
        return {"role": role, "text": _tail(message_text(message.get("content")))}
    parts = [message_text(message.get("content")).strip()]
    if role == "assistant":
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict):
                parts.append(f"-> {function.get('name', '?')}({function.get('arguments', '')})"[:CALL_CHARS])
    return {"role": role, "text": "\n".join(p for p in parts if p)}


def build_state(messages: list[Any]) -> dict[str, Any]:
    convo = [m for m in messages if isinstance(m, dict) and m.get("role") in ROLES] if isinstance(messages, list) else []
    last_user = max((i for i, m in enumerate(convo) if m["role"] == "user"), default=None)
    if last_user is None:
        before, after, user_message = convo[-MAX_CONVERSATION:], [], ""
    else:
        before, after = convo[:last_user][-MAX_CONVERSATION:], convo[last_user + 1:]
        user_message = message_text(convo[last_user].get("content"))
    return {"agent": AGENT, "conversation": [_render(m) for m in before + after], "user_message": user_message}


def _size(state: dict[str, Any]) -> int:
    return len(json.dumps(state, ensure_ascii=False))


def fit_state(state: dict[str, Any], budget: int) -> dict[str, Any] | None:
    fitted = {"agent": state["agent"], "conversation": list(state["conversation"]),
              "user_message": state["user_message"]}
    while _size(fitted) > budget and fitted["conversation"]:
        fitted["conversation"].pop(0)
    if _size(fitted) <= budget:
        return fitted
    text = state["user_message"]
    room = budget - _size({**fitted, "user_message": ""})
    while room > 1:
        head = room // 2
        tail = room - head - 1
        fitted["user_message"] = text[:head] + "…" + (text[len(text) - tail:] if tail else "")
        excess = _size(fitted) - budget
        if excess <= 0:
            return fitted
        room -= excess
    return None
```

- [ ] **Step 4: Run them; expect them to pass**

Run: `$PYTEST tests/test_state.py`
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add plugin/jeff_router/state.py tests/test_state.py
git commit -m "jeff_router: state rendering and budget trimming"
```

---

### Task 5: Comparing Jeff's picks with Strata's action

**Files:**
- Create: `plugin/jeff_router/compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: `options.normalize_path`
- Produces:
  - `strata_action(response) -> dict | None`: `{"kind": "text"}` or `{"kind": "tool", "tools": [names], "first": {"name", "arguments"}}`. It works on both attribute objects and dicts.
  - `pick(answer: dict | None) -> tuple[str | None, float | None]`: the argmax of `probabilities`.
  - `tool_match(pick_key, names, action) -> bool | None`
  - `read_match(pick_key, paths, action, cwd) -> bool | None`

- [ ] **Step 1: Write the failing tests `tests/test_compare.py`**

```python
import json
from types import SimpleNamespace as NS

from jeff_router_under_test.compare import pick, read_match, strata_action, tool_match


def response(content=None, calls=None):
    tool_calls = [NS(id=f"c{i}", type="function", function=NS(name=n, arguments=a)) for i, (n, a) in
                  enumerate(calls or [])] or None
    return NS(choices=[NS(index=0, finish_reason="stop", message=NS(role="assistant", content=content,
                                                                    tool_calls=tool_calls))])


def test_text_response():
    assert strata_action(response(content="hello")) == {"kind": "text"}


def test_tool_response_first_call_parsed():
    action = strata_action(response(calls=[("read_file", json.dumps({"path": "/a"})), ("terminal", "{}")]))
    assert action == {"kind": "tool", "tools": ["read_file", "terminal"],
                      "first": {"name": "read_file", "arguments": {"path": "/a"}}}


def test_dict_shaped_response_and_bad_arguments():
    raw = {"choices": [{"message": {"tool_calls": [{"function": {"name": "x", "arguments": "not json"}}]}}]}
    assert strata_action(raw) == {"kind": "tool", "tools": ["x"], "first": {"name": "x",
                                                                            "arguments": {"_raw": "not json"}}}


def test_missing_choices():
    assert strata_action(NS(choices=[])) is None
    assert strata_action(None) is None


def test_pick_is_argmax():
    assert pick({"probabilities": {"a": 0.2, "b": 0.7, "c": 0.1}}) == ("b", 0.7)
    assert pick(None) == (None, None)
    assert pick({"probabilities": {}}) == (None, None)


def test_tool_match_rules():
    names = {"t1": "read_file", "t2": "terminal"}
    text, tool = {"kind": "text"}, {"kind": "tool", "tools": ["read_file"], "first": {"name": "read_file"}}
    assert tool_match("answer_directly", names, text) is True
    assert tool_match("ask_user", names, text) is True
    assert tool_match("answer_directly", names, tool) is False
    assert tool_match("t1", names, tool) is True
    assert tool_match("t2", names, tool) is False
    assert tool_match("t1", names, text) is False
    assert tool_match(None, names, tool) is None
    assert tool_match("t1", names, None) is None


def test_read_match_rules(tmp_path):
    a = str((tmp_path / "a.md").resolve())
    paths = {"r1": a}
    read = lambda p: {"kind": "tool", "tools": ["read_file"],  # noqa: E731
                      "first": {"name": "read_file", "arguments": {"path": p}}}
    assert read_match("r1", paths, read(a), cwd="") is True
    assert read_match("r1", paths, read("a.md"), cwd=str(tmp_path)) is True
    assert read_match("r1", paths, read("/elsewhere.md"), cwd="") is False
    assert read_match("none_of_these", paths, read("/elsewhere.md"), cwd="") is True
    assert read_match("none_of_these", paths, read(a), cwd="") is False
    assert read_match("r1", paths, {"kind": "text"}, cwd="") is None
    assert read_match("r1", paths, {"kind": "tool", "tools": ["terminal"],
                                    "first": {"name": "terminal", "arguments": {}}}, cwd="") is None
    assert read_match(None, paths, read(a), cwd="") is None
```

- [ ] **Step 2: Run them; expect failure**

Run: `$PYTEST tests/test_compare.py`
Expected: FAIL with `ModuleNotFoundError ... compare`.

- [ ] **Step 3: Implement `plugin/jeff_router/compare.py`**

```python
"""Strata's actual next action, and whether Jeff's picks agree with it."""
from __future__ import annotations

import json
from typing import Any

from .options import normalize_path


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {"_raw": raw[:500]}
        return parsed if isinstance(parsed, dict) else {"_raw": raw[:500]}
    return {}


def strata_action(response: Any) -> dict[str, Any] | None:
    choices = _get(response, "choices") if response is not None else None
    if not choices:
        return None
    message = _get(choices[0], "message")
    calls = _get(message, "tool_calls") or []
    if not calls:
        return {"kind": "text"}
    names = [_get(_get(call, "function"), "name") for call in calls]
    first_function = _get(calls[0], "function")
    return {"kind": "tool", "tools": names,
            "first": {"name": names[0], "arguments": _arguments(_get(first_function, "arguments"))}}


def pick(answer: Any) -> tuple[str | None, float | None]:
    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    if not isinstance(probabilities, dict) or not probabilities:
        return None, None
    key = max(probabilities, key=probabilities.get)
    return key, float(probabilities[key])


def tool_match(pick_key: str | None, names: dict[str, str], action: dict[str, Any] | None) -> bool | None:
    if pick_key is None or action is None:
        return None
    if pick_key in ("answer_directly", "ask_user"):
        return action["kind"] == "text"
    return action["kind"] == "tool" and names.get(pick_key) == action["first"]["name"]


def read_match(pick_key: str | None, paths: dict[str, str], action: dict[str, Any] | None, cwd: str) -> bool | None:
    if pick_key is None or action is None or action["kind"] != "tool" or action["first"]["name"] != "read_file":
        return None
    target = normalize_path(action["first"]["arguments"].get("path"), cwd)
    if pick_key == "none_of_these":
        return target not in paths.values()
    return target is not None and paths.get(pick_key) == target
```

- [ ] **Step 4: Run them; expect them to pass**

Run: `$PYTEST tests/test_compare.py`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add plugin/jeff_router/compare.py tests/test_compare.py
git commit -m "jeff_router: compare Jeff picks with Strata's action"
```

---

### Task 6: Shadow log with rotation

**Files:**
- Create: `plugin/jeff_router/shadow_log.py`
- Test: `tests/test_shadow_log.py`

**Interfaces:**
- Produces: `ShadowLog(path: Path, max_bytes: int)` with `.append(record: dict) -> None`. It writes one line per record with a single `os.write` in append mode. It rotates to `<name>.1` when appending would cross `max_bytes`. Rotate and append run under an exclusive `fcntl.flock` on `<name>.lock`, because the gateway, desktop app and CLI are separate processes sharing the file. It may raise `OSError`; the caller catches it.

- [ ] **Step 1: Write the failing tests `tests/test_shadow_log.py`**

```python
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
```

- [ ] **Step 2: Run them; expect failure**

Run: `$PYTEST tests/test_shadow_log.py`
Expected: FAIL with `ModuleNotFoundError ... shadow_log`.

- [ ] **Step 3: Implement `plugin/jeff_router/shadow_log.py`**

```python
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
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(self.path.with_name(self.path.name + ".lock"), os.O_WRONLY | os.O_CREAT, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                try:
                    if self.path.stat().st_size + len(line) > self.max_bytes:
                        os.replace(self.path, self.path.with_name(self.path.name + ".1"))
                except FileNotFoundError:
                    pass
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                try:
                    os.write(fd, line)
                finally:
                    os.close(fd)
            finally:
                os.close(lock_fd)  # closing the descriptor releases the flock
```

- [ ] **Step 4: Run them; expect them to pass**

Run: `$PYTEST tests/test_shadow_log.py`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add plugin/jeff_router/shadow_log.py tests/test_shadow_log.py
git commit -m "jeff_router: rotating JSONL shadow log"
```

---

### Task 7: The middleware and `register()`

**Files:**
- Create: `plugin/jeff_router/middleware.py`
- Modify: `plugin/jeff_router/__init__.py` (complete `register`)
- Test: `tests/test_middleware.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6 with the exact names above.
- Produces:
  - `ShadowRouter(*, client: JeffClient, log: ShadowLog, cwd: Callable[[], str], max_request_chars: int, slots: int = 2)`, callable as `router(request=None, next_call=None, **context)`. It returns `next_call()`'s result without waiting for Jeff; the record is appended from a done-callback.
  - `register(ctx)` registers `router` under `"llm_execution"`.

- [ ] **Step 1: Write the failing tests `tests/test_middleware.py`**

```python
import json
import logging
import time
from types import SimpleNamespace as NS

import pytest

from conftest import ok_body
from jeff_router_under_test import register
from jeff_router_under_test.jeff_client import JeffClient
from jeff_router_under_test.middleware import ShadowRouter
from jeff_router_under_test.shadow_log import ShadowLog

TOOLS = [{"type": "function", "function": {"name": n, "description": f"{n} tool.",
                                           "parameters": {"required": ["path"]}}}
         for n in ("read_file", "terminal")]
CONTEXT = {"turn_id": "turn-1", "api_call_count": 1, "api_request_id": "req-1", "session_id": "s-1",
           "model": "swift-1.5-iq3_s", "api_mode": "chat_completions"}
HI = {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS}


def strata_text():
    return NS(choices=[NS(index=0, finish_reason="stop", message=NS(role="assistant", content="hi",
                                                                    tool_calls=None))])


def strata_read(path):
    call = NS(id="c1", type="function", function=NS(name="read_file", arguments=json.dumps({"path": path})))
    return NS(choices=[NS(index=0, finish_reason="tool_calls", message=NS(role="assistant", content=None,
                                                                          tool_calls=[call]))])


def make_router(url, tmp_path, *, timeout_s=2.0, max_request_chars=24000, cwd=""):
    log_path = tmp_path / "shadow.jsonl"
    router = ShadowRouter(client=JeffClient(url, timeout_s), log=ShadowLog(log_path, 10_000_000),
                          cwd=lambda: cwd, max_request_chars=max_request_chars)
    return router, log_path


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def wait_records(path, count, timeout=3.0):
    """Records are appended from Jeff's done-callback, so poll until `count` lines exist."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = records(path)
        if len(found) >= count:
            return found
        time.sleep(0.01)
    raise AssertionError(f"expected {count} records, found {len(records(path))}")


class NextCall:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.response


def test_ok_flow_returns_response_untouched_and_logs(fake_jeff, tmp_path):
    target = tmp_path / "notes.md"
    target.write_text("x")
    fake_jeff.reply = lambda body: (200, ok_body({"tool": {"answer_directly": 0.05, "ask_user": 0.05, "t1": 0.8,
                                                           "t2": 0.1},
                                                  "read_target": {"r1": 0.9, "none_of_these": 0.1}}))
    router, log = make_router(fake_jeff.url, tmp_path)
    request = {"model": "m", "messages": [{"role": "user", "content": f"read {target}"}], "tools": TOOLS}
    before = json.dumps(request, sort_keys=True)
    expected = strata_read(str(target))
    nxt = NextCall(expected)
    assert router(request=request, next_call=nxt, **CONTEXT) is expected
    assert nxt.calls == [((), {})]  # called with no payload
    assert json.dumps(request, sort_keys=True) == before
    [rec] = wait_records(log, 1)
    [sent] = fake_jeff.requests
    assert sent["model"] == "tools" and set(sent["questions"]) == {"tool", "read_target"}
    assert rec["v"] == 1 and rec["outcome"] == "ok"
    assert (rec["session_id"], rec["turn_id"], rec["api_request_id"], rec["api_call_count"], rec["model"]) == (
        "s-1", "turn-1", "req-1", 1, "swift-1.5-iq3_s")
    assert rec["tool"]["pick"] == "t1" and rec["tool"]["p"] == 0.8 and rec["tool"]["names"]["t1"] == "read_file"
    assert rec["read_target"]["pick"] == "r1" and rec["read_target"]["candidates"]["r1"] == str(target.resolve())
    assert rec["strata"]["kind"] == "tool" and rec["tool_match"] is True and rec["read_match"] is True
    assert rec["state"]["user_message"] == f"read {target}"
    assert rec["input_tokens"] == 100 and rec["jeff_ms"] >= 0


def test_slow_jeff_never_delays_the_response(fake_jeff, tmp_path):
    fake_jeff.delay = 0.5
    router, log = make_router(fake_jeff.url, tmp_path)
    started = time.monotonic()
    router(request=HI, next_call=NextCall(strata_text()), **CONTEXT)
    assert time.monotonic() - started < 0.2
    assert records(log) == []  # Jeff has not answered yet
    assert wait_records(log, 1)[0]["outcome"] == "ok"


def test_third_concurrent_call_is_local_busy_and_never_queued(fake_jeff, tmp_path):
    fake_jeff.delay = 0.5
    router, log = make_router(fake_jeff.url, tmp_path)
    for count in (1, 2, 3):
        router(request=HI, next_call=NextCall(strata_text()), **{**CONTEXT, "api_call_count": count})
    outcomes = {r["api_call_count"]: r["outcome"] for r in wait_records(log, 3)}
    assert outcomes == {1: "ok", 2: "ok", 3: "local_busy"}
    assert len(fake_jeff.requests) == 2
    # slots are released once Jeff answers: a later call is admitted again
    router(request=HI, next_call=NextCall(strata_text()), **{**CONTEXT, "api_call_count": 4})
    assert {r["api_call_count"]: r["outcome"] for r in wait_records(log, 4)}[4] == "ok"


@pytest.mark.parametrize("context", [
    {**CONTEXT, "turn_id": None},
    {**CONTEXT, "api_call_count": None},
    {**CONTEXT, "api_call_count": True},
    {**CONTEXT, "purpose": "warm_compaction"},
    {**CONTEXT, "api_mode": "codex_responses"},
])
def test_gated_calls_pass_through_without_jeff_or_log(fake_jeff, tmp_path, context):
    router, log = make_router(fake_jeff.url, tmp_path)
    expected = strata_text()
    nxt = NextCall(expected)
    assert router(request=HI, next_call=nxt, **context) is expected
    assert nxt.calls == [((), {})]
    time.sleep(0.1)
    assert fake_jeff.requests == [] and records(log) == []


def test_no_tools_passes_through(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path)
    router(request={"messages": [{"role": "user", "content": "hi"}], "tools": []},
           next_call=NextCall(strata_text()), **CONTEXT)
    time.sleep(0.1)
    assert fake_jeff.requests == [] and records(log) == []


def test_next_call_error_propagates_nothing_is_logged_and_slots_free(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path)
    error = ConnectionError("strata down")
    for _ in range(3):  # more failures than slots: each in-flight Jeff call must free its slot
        with pytest.raises(ConnectionError) as raised:
            router(request=HI, next_call=NextCall(error=error), **CONTEXT)
        assert raised.value is error
        time.sleep(0.1)
    assert records(log) == []
    router(request=HI, next_call=NextCall(strata_text()), **CONTEXT)
    assert wait_records(log, 1)[0]["outcome"] == "ok"


def test_jeff_failure_is_logged_and_response_returned(tmp_path):
    router, log = make_router("http://127.0.0.1:9", tmp_path)
    expected = strata_text()
    assert router(request=HI, next_call=NextCall(expected), **CONTEXT) is expected
    [rec] = wait_records(log, 1)
    assert rec["outcome"] == "refused" and rec["tool"] is None and rec["strata"] == {"kind": "text"}
    assert rec["tool_match"] is None


def test_questions_over_budget_skip_jeff_as_too_long(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path, max_request_chars=50)
    router(request=HI, next_call=NextCall(strata_text()), **CONTEXT)
    assert records(log)[0]["outcome"] == "too_long"  # written at once: Jeff was not asked
    assert fake_jeff.requests == []


def test_build_error_is_logged_not_raised(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path)
    router._cwd = lambda: (_ for _ in ()).throw(RuntimeError("no session"))
    expected = strata_text()
    assert router(request=HI, next_call=NextCall(expected), **CONTEXT) is expected
    assert records(log)[0]["outcome"] == "build_error"


def test_log_failure_never_raises_and_warns_once(fake_jeff, tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    blocker = tmp_path / "file"
    blocker.write_text("not a dir")
    router = ShadowRouter(client=JeffClient(fake_jeff.url, 2.0), log=ShadowLog(blocker / "shadow.jsonl", 1000),
                          cwd=lambda: "", max_request_chars=24000)
    expected = strata_text()
    for _ in range(2):
        assert router(request=HI, next_call=NextCall(expected), **CONTEXT) is expected
        time.sleep(0.2)  # let the callback run and swallow the OSError
    warnings = [r for r in caplog.records if "jeff_router: complete failed" in r.getMessage()]
    assert len(warnings) == 1


def test_register_wires_middleware_and_log_path(tmp_path, fake_jeff):
    registered = []
    settings = {"url": fake_jeff.url}
    ctx = NS(get_config=lambda key, default: settings.get(key, default),
             register_middleware=lambda kind, fn: registered.append((kind, fn)),
             state=NS(data_dir=tmp_path / "plugin-data"), session_cwd=lambda: "")
    register(ctx)
    [(kind, router)] = registered
    assert kind == "llm_execution"
    router(request=HI, next_call=NextCall(strata_text()), **CONTEXT)
    assert wait_records(tmp_path / "plugin-data" / "shadow.jsonl", 1)[0]["outcome"] == "ok"


def test_register_disabled_registers_nothing(tmp_path):
    registered = []
    ctx = NS(get_config=lambda key, default: False if key == "enabled" else default,
             register_middleware=lambda kind, fn: registered.append(kind), state=NS(data_dir=tmp_path))
    register(ctx)
    assert registered == []


def test_close_stops_admission_and_passes_calls_through(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path)
    router.close()
    expected = strata_text()
    assert router(request=HI, next_call=NextCall(expected), **CONTEXT) is expected
    time.sleep(0.1)
    assert fake_jeff.requests == [] and records(log) == []
```

- [ ] **Step 2: Run them; expect failure**

Run: `$PYTEST tests/test_middleware.py`
Expected: FAIL with `ModuleNotFoundError ... middleware`.

- [ ] **Step 3: Implement `plugin/jeff_router/middleware.py`**

```python
"""llm_execution middleware: ask Jeff in the background, pass the call to Strata unchanged, log both.

The agent thread never waits for Jeff: the record is appended from the Jeff future's done-callback. At most `slots`
Jeff requests are in flight per router; a call that finds no free slot is logged as local_busy instead of queueing."""
from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from .compare import pick, read_match, strata_action, tool_match
from .jeff_client import JeffClient, JeffResult
from .options import (
    READ_INSTRUCTIONS, TOOL_INSTRUCTIONS, ReadOptions, ToolOptions, build_read_options, build_tool_options,
)
from .shadow_log import ShadowLog
from .state import build_state, fit_state

logger = logging.getLogger(__name__)
_ENVELOPE_CHARS = 64  # {"model": "tools", "state": , "questions": } around state and questions


@dataclass
class _Pending:
    outcome: str | None = None
    cwd: str = ""
    tool: ToolOptions | None = None
    read: ReadOptions | None = None
    state: dict[str, Any] | None = None
    future: Future | None = None


class ShadowRouter:
    def __init__(self, *, client: JeffClient, log: ShadowLog, cwd: Callable[[], str], max_request_chars: int,
                 slots: int = 2) -> None:
        self._client = client
        self._log = log
        self._cwd = cwd
        self._max_request_chars = max_request_chars
        self._slots = threading.BoundedSemaphore(slots)
        self._executor = ThreadPoolExecutor(max_workers=slots, thread_name_prefix="jeff_router")
        self._warned: set[str] = set()
        self._closed = False

    def __call__(self, request: Any = None, next_call: Any = None, **context: Any) -> Any:
        if self._closed or not self._gated(request, context):
            return next_call()
        pending = self._start(request)
        response = next_call()
        self._finish(pending, response, context)
        return response

    def close(self) -> None:
        """Plugin unload: stop admitting Jeff calls and drop queued work. Running HTTP calls finish and free their
        slots; their records are still written."""
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _gated(request: Any, context: dict[str, Any]) -> bool:
        count = context.get("api_call_count")
        tools = request.get("tools") if isinstance(request, dict) else None
        return (bool(context.get("turn_id")) and isinstance(count, int) and not isinstance(count, bool)
                and context.get("purpose") is None and context.get("api_mode") == "chat_completions"
                and isinstance(tools, list) and bool(tools))

    def _start(self, request: dict[str, Any]) -> _Pending:
        """Runs on the agent thread (session_cwd() is task-local). Never raises."""
        try:
            cwd = self._cwd() or ""
            messages = request.get("messages") or []
            tool = build_tool_options(request["tools"])
            read = build_read_options(messages, set(tool.names.values()), cwd)
            questions: dict[str, Any] = {"tool": {"type": "choice", "instructions": TOOL_INSTRUCTIONS,
                                                  "criteria": tool.criteria}}
            if read is not None:
                questions["read_target"] = {"type": "choice", "instructions": READ_INSTRUCTIONS,
                                            "criteria": read.criteria}
            budget = self._max_request_chars - len(json.dumps(questions, ensure_ascii=False)) - _ENVELOPE_CHARS
            state = fit_state(build_state(messages), budget) if budget > 0 else None
            pending = _Pending(cwd=cwd, tool=tool, read=read, state=state)
            if state is None:
                pending.outcome = "too_long"
                return pending
            if not self._slots.acquire(blocking=False):
                pending.outcome = "local_busy"
                return pending
            try:
                pending.future = self._executor.submit(self._ask, {"model": "tools", "state": state,
                                                                   "questions": questions})
            except BaseException:
                self._slots.release()
                raise
            return pending
        except Exception as error:
            self._warn("build", error)
            return _Pending(outcome="build_error")

    def _ask(self, body: dict[str, Any]) -> JeffResult:
        """Pool thread. The slot covers only the Jeff HTTP call: holding it through Strata's round trip would turn
        concurrent gateway turns into local_busy while Jeff sits idle. Logging afterwards never queues work, because
        the only lock it can meet is another writer's single append (the report reads without locking)."""
        try:
            return self._client.ask(body)
        finally:
            self._slots.release()

    def _finish(self, pending: _Pending, response: Any, context: dict[str, Any]) -> None:
        """Runs on the agent thread after Strata answered. Never waits for Jeff, never raises."""
        try:
            action = strata_action(response)
        except Exception as error:
            self._warn("finish", error)
            action = None
        if pending.future is not None:
            pending.future.add_done_callback(lambda future: self._complete(pending, future, action, context))
            return
        try:
            self._log.append(self._record(pending, pending.outcome or "bad_response", None, action, context))
        except Exception as error:
            self._warn("finish", error)

    def _complete(self, pending: _Pending, future: Future, action: dict[str, Any] | None,
                  context: dict[str, Any]) -> None:
        """Done-callback: runs in the pool thread when Jeff finishes, or inline if it already has. Appending is one
        short write under the writers' flock (readers take no lock), so running inline does not stall the turn."""
        if future.cancelled():  # dropped by close(); nothing to compare
            return
        try:
            try:
                result: JeffResult | None = future.result()
                outcome = result.outcome
            except Exception:
                result, outcome = None, "bad_response"
            self._log.append(self._record(pending, outcome, result, action, context))
        except Exception as error:
            self._warn("complete", error)

    @staticmethod
    def _record(pending: _Pending, outcome: str, result: JeffResult | None, action: dict[str, Any] | None,
                context: dict[str, Any]) -> dict[str, Any]:
        ok = outcome == "ok" and result is not None and result.answers is not None
        tool_block = read_block = None
        tool_ok = read_ok = None
        if ok and pending.tool is not None:
            key, p = pick(result.answers.get("tool"))
            tool_block = {"options": pending.tool.criteria, "names": pending.tool.names,
                          "probabilities": result.answers["tool"]["probabilities"], "pick": key, "p": p,
                          "truncated_tools": pending.tool.truncated}
            tool_ok = tool_match(key, pending.tool.names, action)
            if pending.read is not None and "read_target" in result.answers:
                rkey, rp = pick(result.answers["read_target"])
                read_block = {"candidates": pending.read.paths,
                              "probabilities": result.answers["read_target"]["probabilities"], "pick": rkey, "p": rp}
                read_ok = read_match(rkey, pending.read.paths, action, pending.cwd)
        return {
            "v": 1,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "session_id": context.get("session_id"), "turn_id": context.get("turn_id"),
            "api_request_id": context.get("api_request_id"), "api_call_count": context.get("api_call_count"),
            "model": context.get("model"), "outcome": outcome,
            "jeff_ms": result.ms if result else None, "input_tokens": result.input_tokens if ok else None,
            "tool": tool_block, "read_target": read_block, "strata": action,
            "tool_match": tool_ok, "read_match": read_ok, "state": pending.state,
        }

    def _warn(self, where: str, error: Exception) -> None:
        key = f"{where}:{type(error).__name__}"
        if key in self._warned:
            logger.debug("jeff_router: %s failed again: %s", where, error)
            return
        self._warned.add(key)
        logger.warning("jeff_router: %s failed (%s: %s); the call continues unchanged and later errors of this "
                       "kind are logged at DEBUG", where, type(error).__name__, error)
```

- [ ] **Step 4: Complete `plugin/jeff_router/__init__.py`**

```python
"""jeff_router: shadow-mode Jeff router for Hermes Agent (see docs/superpowers/specs in the jeff-hermes repo)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _cwd_reader(ctx: Any) -> Callable[[], str]:
    session_cwd = getattr(ctx, "session_cwd", None)
    if not callable(session_cwd):
        return lambda: ""
    return lambda: session_cwd() or ""


def register(ctx: Any) -> None:
    """Hermes entry point: register the shadow router as llm_execution middleware."""
    from .jeff_client import JeffClient
    from .middleware import ShadowRouter
    from .settings import read_settings
    from .shadow_log import ShadowLog

    settings = read_settings(ctx.get_config)
    if not settings["enabled"]:
        logger.info("jeff_router: disabled in settings; registering nothing")
        return
    log = ShadowLog(Path(ctx.state.data_dir) / "shadow.jsonl", max_bytes=int(settings["log_max_mb"] * 1024 * 1024))
    router = ShadowRouter(client=JeffClient(settings["url"], settings["timeout_s"]), log=log, cwd=_cwd_reader(ctx),
                          max_request_chars=int(settings["max_request_chars"]))
    ctx.register_middleware("llm_execution", router)
    on_unload = getattr(ctx, "on_unload", None)
    if callable(on_unload):
        on_unload(router.close)
    logger.info("jeff_router: shadow mode on; Jeff at %s, log %s", settings["url"], log.path)


__all__ = ["register"]
```

- [ ] **Step 5: Run the middleware tests, then the whole suite; expect all to pass**

Run: `$PYTEST tests/test_middleware.py && $PYTEST`
Expected: `17 passed` for the first run; the full suite reports every test passed (5+10+13+10+7+5+17 = 67).

- [ ] **Step 6: Commit**

```bash
git add plugin/jeff_router/middleware.py plugin/jeff_router/__init__.py tests/test_middleware.py
git commit -m "jeff_router: shadow middleware and register()"
```

---

### Task 8: Report script

**Files:**
- Create: `tools/shadow_report.py`
- Test: `tests/test_shadow_report.py`

**Interfaces:**
- Consumes: the record format from Task 7, and `READ_ONLY_TOOLS` from `plugin/jeff_router/options.py`, loaded by file path.
- Produces:
  - `load(paths: list[str]) -> list[dict]`: skips bad lines.
  - `summarize(records: list[dict]) -> str`: Markdown.
  - CLI `python tools/shadow_report.py [paths…]`

- [ ] **Step 1: Write the failing tests `tests/test_shadow_report.py`**

```python
import importlib.util
import json
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
    # t=0.9: 3 of 5 ok records covered, all three agree with Strata
    assert "| 0.9 | 60.0% (3) | 100.0% |" in text
    # t=0.7: 4 covered, 3 agree
    assert "| 0.7 | 80.0% (4) | 75.0% |" in text
    # full read_file call at t=0.5: two calls Jeff would have answered, one match, one mismatch
    assert "| 0.5 | 2 | 1 | 1 |" in text
    assert "accuracy" not in text and "correct" not in text  # Strata is a reference, not ground truth


def test_empty_input():
    assert "No records" in report.summarize([])
```

- [ ] **Step 2: Run them; expect failure**

Run: `$PYTEST tests/test_shadow_report.py`
Expected: FAIL with `FileNotFoundError ... tools/shadow_report.py`.

- [ ] **Step 3: Implement `tools/shadow_report.py`**

```python
#!/usr/bin/env python3
"""Summarize jeff_router shadow logs: how often Jeff agreed with Strata's actual next action, by threshold.

Strata's action is a reference, not ground truth, so the report speaks of agreement, never accuracy. Jeff's
answer_directly and ask_user both count as agreeing with any Strata text reply.

Usage: python tools/shadow_report.py [path ...]
With no paths it reads ~/.hermes/plugin-data/jeff_router/shadow.jsonl.1 and shadow.jsonl without taking the
writers' lock (so it never blocks Hermes); if shadow.jsonl.1 is replaced while reading, it reads both again."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

_OPTIONS = Path(__file__).resolve().parents[1] / "plugin" / "jeff_router" / "options.py"
_spec = importlib.util.spec_from_file_location("jeff_router_options", _OPTIONS)
_options = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_options)
READ_ONLY_TOOLS = _options.READ_ONLY_TOOLS
THRESHOLDS = [round(0.3 + 0.1 * i, 1) for i in range(7)]
DEFAULT_LOG = Path.home() / ".hermes" / "plugin-data" / "jeff_router" / "shadow.jsonl"


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


def load_default(log: Path = DEFAULT_LOG) -> list[dict]:
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


def summarize(records: list[dict]) -> str:
    if not records:
        return "No records."
    lines = ["## Counts", "", "| | count | share |", "|---|---:|---:|", f"| records | {len(records)} |"]
    outcomes = Counter(r.get("outcome") for r in records)
    lines += [f"| {o} | {n} | {_pct(n, len(records))} |" for o, n in outcomes.most_common()]
    ok = [r for r in records if r.get("outcome") == "ok" and r.get("tool") and r.get("tool_match") is not None]
    ms = [r["jeff_ms"] for r in ok if isinstance(r.get("jeff_ms"), (int, float))]
    kinds = Counter((r.get("strata") or {}).get("kind") for r in records if r.get("strata"))
    lines += ["", f"Jeff latency (ok): p50 {_percentile(ms, 0.5)} ms, p95 {_percentile(ms, 0.95)} ms. "
              f"Strata actions: tool {kinds.get('tool', 0)}, text {kinds.get('text', 0)}.", ""]

    lines += ["## Tool choice: agreement with Strata", "",
              "| threshold | coverage | agreement | read-only coverage | read-only agreement |",
              "|---:|---:|---:|---:|---:|"]
    for t in THRESHOLDS:
        covered = [r for r in ok if r["tool"]["p"] >= t]
        read_only = [r for r in covered if _picked_name(r) in READ_ONLY_TOOLS]
        lines.append(f"| {t} | {_pct(len(covered), len(ok))} ({len(covered)}) | "
                     f"{_pct(sum(r['tool_match'] for r in covered), len(covered))} | "
                     f"{_pct(len(read_only), len(ok))} ({len(read_only)}) | "
                     f"{_pct(sum(r['tool_match'] for r in read_only), len(read_only))} |")

    full = [r for r in ok if _picked_name(r) == "read_file" and r.get("read_target")
            and str(r["read_target"].get("pick", "")).startswith("r") and r.get("read_match") is not None]
    lines += ["", "## Full read_file call (tool p and read_target p both ≥ threshold)", "",
              "| threshold | Jeff would answer | match | mismatch |", "|---:|---:|---:|---:|"]
    for t in THRESHOLDS:
        covered = [r for r in full if r["tool"]["p"] >= t and r["read_target"]["p"] >= t]
        matched = sum(1 for r in covered if r["read_match"])
        lines.append(f"| {t} | {len(covered)} | {matched} | {len(covered) - matched} |")
    lines += ["", f"Strata calls in the log: {sum(kinds.values())}. A match is a call Jeff could have answered "
              "the way Strata did. Mismatches are not necessarily Jeff errors: review a sample of high-confidence "
              "mismatches by hand before trusting any threshold."]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    records = load(argv) if argv else load_default()
    if not records:
        print("No records found." if argv else f"No records in {DEFAULT_LOG}(.1).")
        return 1
    print(summarize(records))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run them; expect them to pass**

Run: `$PYTEST tests/test_shadow_report.py`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add tools/shadow_report.py tests/test_shadow_report.py
git commit -m "Shadow report: Jeff vs Strata agreement by threshold"
```

---

### Task 9: `jeff-serve` systemd user unit

**Files:**
- Create: `deploy/jeff-serve.service`
- Install: `~/.config/systemd/user/jeff-serve.service`

**Interfaces:**
- Produces: jeff-serve listening on `127.0.0.1:8765`, with `GET /health` → `{"status": "ready", ...}` and the adapters `tools`, `code-router` and `triage` loaded.

- [ ] **Step 0: Look up the systemd semantics with Context7 (required by `~/.omp/agent/AGENTS.md`)**

Check the installed version first: `systemctl --version | head -1`. Then resolve `systemd` through Context7
(`xd://mcp__context7_resolve_library_id`) and query the docs (`xd://mcp__context7_query_docs`) on these points:
- user units under `~/.config/systemd/user` and `WantedBy=default.target` with lingering;
- `Restart=on-failure` and `RestartSec`, including whether `SIGKILL` from `systemctl kill` counts as a failure;
- multiple `Environment=` lines;
- `WorkingDirectory` with relative paths in the environment.

If the docs contradict the unit below, adjust the unit and note the source in the commit message.

- [ ] **Step 1: Create `deploy/jeff-serve.service`**

```ini
[Unit]
Description=Jeff decision model (jeff-serve) on the GPU for the Hermes jeff_router plugin (:8765)
After=default.target

[Service]
WorkingDirectory=/path/to/jeff-hermes
Environment=JEFF_DEVICE=cuda
Environment=JEFF_CHECKPOINT=models/jeff-base-v1.3
Environment=JEFF_ADAPTERS=models/adapters
Environment=JEFF_HOST=127.0.0.1
Environment=PORT=8765
Environment=JEFF_QUEUE_MS=250
ExecStart=/path/to/jeff-hermes/.venv-gpu/bin/jeff-serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

- [ ] **Step 2: Check that nothing else holds port 8765**

Run: `ss -ltn | grep -c ':8765 '`
Expected: `0`. If it prints 1, stop the process holding the port; it is a leftover manual `jeff-serve`.

- [ ] **Step 3: Install, enable and start it**

```bash
cp deploy/jeff-serve.service ~/.config/systemd/user/jeff-serve.service
systemctl --user daemon-reload
systemctl --user enable --now jeff-serve.service
```

- [ ] **Step 4: Verify that it is up, on the GPU, with the adapters**

Run: `for i in $(seq 1 30); do curl -sf 127.0.0.1:8765/health && break; sleep 1; done; echo; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | grep -i python`
Expected:
- JSON with `"status":"ready"` and adapters `code-router`, `triage` and `tools`.
- One python process at about 2.2 GB.

Run: `curl -s -H 'Content-Type: application/json' -d @models/adapters/tools/example.json 127.0.0.1:8765/v1/systemone | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['answers']['tool']['choice'], d['usage']['input_tokens'])"`
Expected: a choice key such as `t3`, and a token count of about 363.

- [ ] **Step 5: Check that it restarts on failure**

Run: `systemctl --user kill -s KILL jeff-serve.service; sleep 12; systemctl --user is-active jeff-serve.service; curl -sf 127.0.0.1:8765/health >/dev/null && echo healthy`
Expected: `active` then `healthy`.

- [ ] **Step 6: Commit**

```bash
git add deploy/jeff-serve.service
git commit -m "deploy: jeff-serve systemd user unit (GPU, :8765)"
```

---

### Task 10: End-to-end probe with a real `AIAgent` and the real jeff-serve

**Files:**
- Create: `probes/shadow_probe.py`

**Interfaces:**
- Consumes:
  - `register(ctx)` from the plugin, loaded by file path as package `jeff_router`.
  - The running jeff-serve from Task 9.
  - The Hermes runtime's `_delivery_manager()._middleware`, as in `probes/short_circuit_probe.py`.

- [ ] **Step 1: Create `probes/shadow_probe.py`**

```python
"""End-to-end check: jeff_router in a real AIAgent turn. A fake provider middleware after jeff_router plays
Strata (no network): call 1 returns a read_file tool call, call 2 returns final text. Asserts one ok shadow record
per call, the right match flags, and an unchanged transcript. Needs jeff-serve on :8765."""
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace as NS

from hermes_cli.plugins import _delivery_manager
from run_agent import AIAgent

REPO = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO / "plugin" / "jeff_router"
TARGET = REPO / "docs/research/01-hook-api.md"

spec = importlib.util.spec_from_file_location("jeff_router", PLUGIN_DIR / "__init__.py",
                                              submodule_search_locations=[str(PLUGIN_DIR)])
plugin = importlib.util.module_from_spec(spec)
sys.modules["jeff_router"] = plugin
spec.loader.exec_module(plugin)

data_dir = Path(tempfile.mkdtemp(prefix="jeff_router_probe_"))
registered = []
ctx = NS(get_config=lambda key, default: default, register_middleware=lambda kind, fn: registered.append((kind, fn)),
         state=NS(data_dir=data_dir), session_cwd=lambda: str(REPO))
plugin.register(ctx)
[(kind, router)] = registered
assert kind == "llm_execution"


def fake_response(message, finish):
    return NS(id="strata-fake", model="strata-fake", object="chat.completion", usage=None,
              choices=[NS(index=0, message=message, finish_reason=finish)])


def fake_strata(request, next_call, api_call_count=None, **ctx):
    if request["messages"][-1]["role"] == "user":
        call = NS(id="call_1", type="function",
                  function=NS(name="read_file", arguments=json.dumps({"path": str(TARGET)})))
        return fake_response(NS(role="assistant", content=None, tool_calls=[call]), "tool_calls")
    return fake_response(NS(role="assistant", content="FAKE-FINAL", tool_calls=None), "stop")


chain = _delivery_manager()._middleware.setdefault("llm_execution", [])
chain.append(router)
chain.append(fake_strata)

agent = AIAgent(base_url="http://127.0.0.1:9/v1", model="probe-model", api_key="dummy",
                enabled_toolsets=["file"], max_iterations=5, quiet_mode=True)
result = agent.run_conversation(f"What does {TARGET} say about warm compaction?")
log_path = data_dir / "shadow.jsonl"
deadline = time.monotonic() + 5  # records are appended from Jeff's done-callback
while time.monotonic() < deadline and (not log_path.exists() or len(log_path.read_text().splitlines()) < 2):
    time.sleep(0.05)
records = sorted((json.loads(line) for line in log_path.read_text().splitlines()), key=lambda r: r["api_call_count"])
for r in records:
    print("RECORD", r["api_call_count"], r["outcome"], r["jeff_ms"], r["input_tokens"],
          r["tool"] and (r["tool"]["names"].get(r["tool"]["pick"], r["tool"]["pick"]), round(r["tool"]["p"], 2)),
          r["read_target"] and (r["read_target"]["pick"], round(r["read_target"]["p"], 2)),
          r["strata"], r["tool_match"], r["read_match"])

assert result["completed"] and result["api_calls"] == 2, result
assert result["final_response"] == "FAKE-FINAL"
assert [m["role"] for m in result["messages"]] == ["user", "assistant", "tool", "assistant"]
assert len(records) == 2, records
assert all(r["outcome"] == "ok" for r in records), [r["outcome"] for r in records]
assert records[0]["strata"]["kind"] == "tool" and records[1]["strata"] == {"kind": "text"}
assert str(TARGET) in records[0]["read_target"]["candidates"].values()
assert records[1]["read_target"] is None or str(TARGET) not in records[1]["read_target"]["candidates"].values()
assert records[0]["tool_match"] in (True, False) and records[1]["tool_match"] in (True, False)
print("SHADOW PROBE OK", data_dir)
```

- [ ] **Step 2: Run it with the Hermes runtime and an isolated `HERMES_HOME`**

```bash
P=$PWD/probes/shadow_probe.py
cd /tmp && HERMES_HOME=$(mktemp -d) ~/.hermes/tools/python-3.14.9+20261021-linux-x64/bin/python3 -I -c \
  "import sys,runpy; sys.path.insert(0,'$HOME/.hermes/hermes-agent'); import hermes_bootstrap; \
   runpy.run_path('$P', run_name='__main__')"
```

Expected:
- Two `RECORD` lines, both `ok`, with `jeff_ms` of about 20–60.
- The first record has a `read_target` with `docs/research/01-hook-api.md` among its candidates.
- The last line is `SHADOW PROBE OK <dir>`.
- Jeff's picks are printed; whether they match is data, not a pass condition.

- [ ] **Step 3: Confirm the Hermes checkout is untouched**

Run: `git -C ~/.hermes/hermes-agent status --short | wc -l`
Expected: `0`.

- [ ] **Step 4: Commit**

```bash
git add probes/shadow_probe.py
git commit -m "probe: end-to-end shadow check with a real AIAgent and jeff-serve"
```

---

### Task 11: Go live, README

**Files:**
- Create: `README.md`
- Modify: `~/.hermes/config.yaml` (`plugins.enabled`, outside the repo)
- Create: symlink `~/.hermes/plugins/jeff_router` → `<repo>/plugin/jeff_router`

- [ ] **Step 1: Symlink the plugin**

Run: `ln -s /path/to/jeff-hermes/plugin/jeff_router ~/.hermes/plugins/jeff_router && ls -l ~/.hermes/plugins/jeff_router`
Expected: `jeff_router -> /path/to/jeff-hermes/plugin/jeff_router`.

- [ ] **Step 2: Add `jeff_router` to `plugins.enabled` in `~/.hermes/config.yaml`**

Read the `plugins:` block first (`~/.hermes/config.yaml:224-235`), then insert `    - jeff_router` as a list item under `enabled:` with the `edit` tool, keeping the existing items and their indentation. Afterwards:

Run: `grep -n -A12 '^plugins:' ~/.hermes/config.yaml`
Expected: `- jeff_router` listed under `enabled:`.

- [ ] **Step 3: Validate the plugin with Hermes' own validator, if the CLI provides one**

Run: `~/.local/bin/hermes plugins validate ~/.hermes/plugins/jeff_router 2>&1 | tail -20`
Expected: no errors about `provides_middleware` or the manifest. If the subcommand does not exist, note that and continue; Task 10 already loaded the plugin through Hermes' module conventions.

- [ ] **Step 4: Ask the user before restarting anything**

Ask: "Restart `hermes-gateway.service` now to load jeff_router? The desktop app and open CLI sessions load it on their next start." Restart only on a yes:

```bash
systemctl --user restart hermes-gateway.service && sleep 5 && systemctl --user is-active hermes-gateway.service
grep -h "jeff_router" ~/.hermes/logs/agent.log | tail -3
```

Expected: `active`, and a log line `jeff_router: shadow mode on; Jeff at http://127.0.0.1:8765, log …/plugin-data/jeff_router/shadow.jsonl`.

- [ ] **Step 5: Check one real turn**

After the user (or a quick CLI turn, if the user agrees) sends one message through a Hermes surface that has loaded the plugin:

Run: `tail -n 1 ~/.hermes/plugin-data/jeff_router/shadow.jsonl | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); print(r['outcome'], r['jeff_ms'], r['strata'], r['tool_match'])"`
Expected: `ok`, a latency of tens of ms, Strata's action, and a match flag.

Run: `.venv/bin/python tools/shadow_report.py`
Expected: a Markdown report with at least one record.

- [ ] **Step 6: Create `README.md`**

````markdown
# jeff-hermes

Jeff (a 0.8B option-scoring model) beside Hermes Agent's big model (Strata). v1 is **shadow mode**: on every
main-turn model call the `jeff_router` plugin asks Jeff what the agent will do next, lets the call go to Strata
unchanged, and logs both. The log decides whether and where v2 lets Jeff answer calls itself.

- Design: `docs/design/2026-10-30-shadow-design.md`
- Research: `docs/research/01-hook-api.md` (hook point), `docs/research/02-install-and-latency.md` (install, latency, coexistence with Hindsight)

## Service

`jeff-serve` runs on the GPU as a systemd user unit (`deploy/jeff-serve.service`, port 8765, about 2.2 GB VRAM):

```sh
systemctl --user status jeff-serve      # restart / stop / disable as usual
curl -s 127.0.0.1:8765/health
```

Rebuilding `.venv-gpu`, the models, and `causal-conv1d`: see `docs/research/02-install-and-latency.md`.

## Plugin

Source: `plugin/jeff_router/`, symlinked as `~/.hermes/plugins/jeff_router` and listed in `plugins.enabled` in
`~/.hermes/config.yaml`. Settings (optional) go under `plugins.entries.jeff_router.settings`: `enabled`, `url`,
`timeout_s`, `max_request_chars`, `log_max_mb`. Turning it off: set `enabled: false`, or remove it
from `plugins.enabled`, then restart Hermes. If jeff-serve is down, the plugin logs `refused` and Hermes is
unaffected.

Log: `~/.hermes/plugin-data/jeff_router/shadow.jsonl` (rotated to `.1`). It contains paths and transcript snippets.

## Report

```sh
.venv/bin/python tools/shadow_report.py            # all of shadow.jsonl*
```

## Tests

```sh
PYTEST="uvx -q --python $HOME/.hermes/tools/python-3.14.9+20261021-linux-x64/bin/python3 --from pytest>=8 pytest"
$PYTEST                                            # unit tests, on the Hermes runtime's Python
```

End-to-end probe (needs jeff-serve): see the command in `docs/design/2026-10-30-shadow-plan.md`,
Task 10.
````

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "README: service, plugin, report, tests"
```

- [ ] **Step 8: Final checks**

Run: `$PYTEST && git -C ~/.hermes/hermes-agent status --short | wc -l && git status --short`
Expected: every test passes; `0`; a clean working tree.
````
