import socket
import threading
import urllib.request

import pytest

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


def _raw_server(response: bytes) -> str:
    """One-shot server: reads a full request, writes `response` verbatim, then closes."""
    listener = socket.create_server(("127.0.0.1", 0))

    def serve():
        with listener, listener.accept()[0] as conn:
            data = b""
            while b"\r\n\r\n" not in data:
                data += conn.recv(65536)
            head, _, rest = data.partition(b"\r\n\r\n")
            length = next(int(line.split(b":", 1)[1]) for line in head.split(b"\r\n")
                          if line.lower().startswith(b"content-length:"))
            while len(rest) < length:
                rest += conn.recv(65536)
            conn.sendall(response)
            conn.shutdown(socket.SHUT_WR)

    threading.Thread(target=serve, daemon=True).start()
    return f"http://127.0.0.1:{listener.getsockname()[1]}"


@pytest.mark.parametrize("response", [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{\"answers\"",
    b"garbage\r\n\r\n",
], ids=["incomplete_read", "bad_status_line"])
def test_broken_http_response_is_server_error(response):
    result = JeffClient(_raw_server(response), 2.0).ask(BODY)
    assert result.outcome == "server_error"


def test_proxy_env_vars_are_ignored(fake_jeff, monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead_proxy = f"http://127.0.0.1:{s.getsockname()[1]}"
    for name in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(name, dead_proxy)
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    # urlopen() caches a global opener whose proxies are read on first use; reset it so this
    # test does not depend on test order.
    monkeypatch.setattr(urllib.request, "_opener", None)
    assert JeffClient(fake_jeff.url, 2.0).ask(BODY).outcome == "ok"
    assert fake_jeff.requests == [BODY]


def test_non_serializable_body_is_bad_request(fake_jeff):
    result = JeffClient(fake_jeff.url, 2.0).ask({**BODY, "state": {"user_message": object()}})
    assert result.outcome == "bad_request"
    assert fake_jeff.requests == []
