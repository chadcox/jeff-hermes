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


class _TestServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def handle_error(self, request, client_address):
        # A client that gave up (e.g. the timeout test) closes before the reply is written;
        # that is expected, not a server failure. Anything else is still reported.
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


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

    server = _TestServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()
