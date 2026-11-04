import json
import logging
import time
from types import SimpleNamespace as NS

import pytest

from conftest import ok_body
from jeff_router_under_test import register
from jeff_router_under_test.compare import MAX_ARGUMENT_CHARS
from jeff_router_under_test.jeff_client import JeffClient
from jeff_router_under_test.middleware import ShadowRouter
from jeff_router_under_test.shadow_log import ShadowLog

TOOLS = [{"type": "function", "function": {"name": n, "description": f"{n} tool.",
                                           "parameters": {"required": ["path"]}}}
         for n in ("read_file", "terminal")]
CONTEXT = {"turn_id": "turn-1", "api_call_count": 1, "api_request_id": "req-1", "session_id": "s-1",
           "model": "swift-1.5-iq3_s", "api_mode": "chat_completions"}
HI = {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS}


def main_text():
    return NS(choices=[NS(index=0, finish_reason="stop", message=NS(role="assistant", content="hi",
                                                                    tool_calls=None))])


def main_read(path):
    call = NS(id="c1", type="function", function=NS(name="read_file", arguments=json.dumps({"path": path})))
    return NS(choices=[NS(index=0, finish_reason="tool_calls", message=NS(role="assistant", content=None,
                                                                          tool_calls=[call]))])


def make_router(url, tmp_path, *, timeout_s=2.0, max_request_chars=24000, cwd="", main_thinking=True):
    log_path = tmp_path / "shadow.jsonl"
    router = ShadowRouter(client=JeffClient(url, timeout_s), log=ShadowLog(log_path, 10_000_000),
                          cwd=lambda: cwd, max_request_chars=max_request_chars, main_thinking=main_thinking)
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
        # Hermes makes next_call single-use per middleware frame
        assert not self.calls, "next_call() called more than once"
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
    expected = main_read(str(target))
    nxt = NextCall(expected)
    assert router(request=request, next_call=nxt, **CONTEXT) is expected
    assert nxt.calls == [((), {})]  # called with no payload
    assert json.dumps(request, sort_keys=True) == before
    [rec] = wait_records(log, 1)
    [sent] = fake_jeff.requests
    assert sent["model"] == "tools" and set(sent["questions"]) == {"tool", "read_target"}
    assert rec["v"] == 2 and rec["outcome"] == "ok"
    assert (rec["session_id"], rec["turn_id"], rec["api_request_id"], rec["api_call_count"], rec["model"]) == (
        "s-1", "turn-1", "req-1", 1, "swift-1.5-iq3_s")
    assert rec["tool"]["pick"] == "t1" and rec["tool"]["p"] == 0.8 and rec["tool"]["names"]["t1"] == "read_file"
    assert rec["read_target"]["pick"] == "r1" and rec["read_target"]["candidates"]["r1"] == str(target.resolve())
    assert rec["main"]["kind"] == "tool" and rec["tool_match"] is True and rec["read_match"] is True
    assert rec["state"]["user_message"] == f"read {target}"
    assert rec["input_tokens"] == 100 and rec["jeff_ms"] >= 0
    assert rec["main_thinking"] is True


def test_long_read_path_matches_uncapped_and_is_logged_capped(fake_jeff, tmp_path):
    base = tmp_path.resolve()
    if len(str(base)) > 2300:
        pytest.skip(f"tmp_path is {len(str(base))} chars; a 2500-char path needs room under it")
    remaining = 2500 - len(str(base))  # each component adds "/" + name
    folder = base
    while remaining > 206:  # leaves 6..206, so the file name is 5..205 chars (under NAME_MAX 255)
        folder = folder / ("d" * 200)
        remaining -= 201
    target = folder / ("f" * (remaining - 1))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x")
    assert len(str(target)) == 2500
    fake_jeff.reply = lambda body: (200, ok_body({"tool": {"answer_directly": 0.05, "ask_user": 0.05, "t1": 0.8,
                                                           "t2": 0.1},
                                                  "read_target": {"r1": 0.9, "none_of_these": 0.1}}))
    router, log = make_router(fake_jeff.url, tmp_path)
    request = {"model": "m", "messages": [{"role": "user", "content": f"read {target}"}], "tools": TOOLS}
    router(request=request, next_call=NextCall(main_read(str(target))), **CONTEXT)
    [rec] = wait_records(log, 1)
    assert rec["read_target"]["candidates"]["r1"] == str(target)
    assert rec["read_match"] is True and rec["tool_match"] is True
    assert rec["main"]["first"]["arguments"]["path"] == str(target)[:MAX_ARGUMENT_CHARS] + "…"


def test_slow_jeff_never_delays_the_response(fake_jeff, tmp_path):
    fake_jeff.delay = 0.5
    router, log = make_router(fake_jeff.url, tmp_path)
    started = time.monotonic()
    router(request=HI, next_call=NextCall(main_text()), **CONTEXT)
    assert time.monotonic() - started < 0.2
    assert records(log) == []  # Jeff has not answered yet
    assert wait_records(log, 1)[0]["outcome"] == "ok"


def test_third_concurrent_call_is_local_busy_and_never_queued(fake_jeff, tmp_path):
    fake_jeff.delay = 0.5
    router, log = make_router(fake_jeff.url, tmp_path)
    for count in (1, 2, 3):
        router(request=HI, next_call=NextCall(main_text()), **{**CONTEXT, "api_call_count": count})
    outcomes = {r["api_call_count"]: r["outcome"] for r in wait_records(log, 3)}
    assert outcomes == {1: "ok", 2: "ok", 3: "local_busy"}
    assert len(fake_jeff.requests) == 2
    # slots are released once Jeff answers: a later call is admitted again
    router(request=HI, next_call=NextCall(main_text()), **{**CONTEXT, "api_call_count": 4})
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
    expected = main_text()
    nxt = NextCall(expected)
    assert router(request=HI, next_call=nxt, **context) is expected
    assert nxt.calls == [((), {})]
    time.sleep(0.1)
    assert fake_jeff.requests == [] and records(log) == []


def test_no_tools_passes_through(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path)
    router(request={"messages": [{"role": "user", "content": "hi"}], "tools": []},
           next_call=NextCall(main_text()), **CONTEXT)
    time.sleep(0.1)
    assert fake_jeff.requests == [] and records(log) == []


def test_next_call_error_propagates_nothing_is_logged_and_slots_free(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path)
    error = ConnectionError("model down")
    for _ in range(3):  # more failures than slots: each in-flight Jeff call must free its slot
        with pytest.raises(ConnectionError) as raised:
            router(request=HI, next_call=NextCall(error=error), **CONTEXT)
        assert raised.value is error
        time.sleep(0.1)
    assert records(log) == []
    router(request=HI, next_call=NextCall(main_text()), **CONTEXT)
    assert wait_records(log, 1)[0]["outcome"] == "ok"


def test_jeff_failure_is_logged_and_response_returned(tmp_path):
    router, log = make_router("http://127.0.0.1:9", tmp_path)
    expected = main_text()
    assert router(request=HI, next_call=NextCall(expected), **CONTEXT) is expected
    [rec] = wait_records(log, 1)
    assert rec["outcome"] == "refused" and rec["tool"] is None and rec["main"] == {"kind": "text"}
    assert rec["tool_match"] is None


def test_main_cost_is_logged(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path)
    response = NS(choices=[NS(index=0, finish_reason="stop",
                              message=NS(role="assistant", content="hello", reasoning_content="thinking...",
                                         tool_calls=None))],
                  usage=NS(prompt_tokens=40, completion_tokens=12))
    request = {**HI, "extra_body": {"reasoning": {"enabled": True, "effort": "high"}}}
    router(request=request, next_call=NextCall(response), **CONTEXT)
    [rec] = wait_records(log, 1)
    cost = rec["main_cost"]
    assert cost["ms"] >= 0
    assert {k: v for k, v in cost.items() if k != "ms"} == {
        "effort": {"enabled": True, "effort": "high"}, "finish_reason": "stop",
        "prompt_tokens": 40, "completion_tokens": 12, "reasoning_chars": 11, "content_chars": 5}


def test_main_cost_without_usage_or_reasoning_is_null_and_logged_when_jeff_fails(tmp_path):
    router, log = make_router("http://127.0.0.1:9", tmp_path)
    router(request=HI, next_call=NextCall(main_read("/x")), **CONTEXT)
    [rec] = wait_records(log, 1)
    assert rec["outcome"] == "refused"
    assert {k: v for k, v in rec["main_cost"].items() if k != "ms"} == {
        "effort": None, "finish_reason": "tool_calls", "prompt_tokens": None, "completion_tokens": None,
        "reasoning_chars": 0, "content_chars": 0}


def test_questions_over_budget_skip_jeff_as_too_long(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path, max_request_chars=50)
    router(request=HI, next_call=NextCall(main_text()), **CONTEXT)
    assert records(log)[0]["outcome"] == "too_long"  # written at once: Jeff was not asked
    assert fake_jeff.requests == []


def test_ok_without_tool_answer_is_bad_response(fake_jeff, tmp_path):
    fake_jeff.reply = (200, ok_body({"read_target": {"r1": 0.9, "none_of_these": 0.1}}))
    router, log = make_router(fake_jeff.url, tmp_path)
    router(request=HI, next_call=NextCall(main_text()), **CONTEXT)
    [rec] = wait_records(log, 1)
    assert rec["outcome"] == "bad_response" and rec["main"] == {"kind": "text"}
    assert (rec["tool"], rec["read_target"], rec["tool_match"], rec["read_match"], rec["input_tokens"]) == (
        None, None, None, None, None)


def test_build_error_is_logged_not_raised(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path)
    router._cwd = lambda: (_ for _ in ()).throw(RuntimeError("no session"))
    expected = main_text()
    assert router(request=HI, next_call=NextCall(expected), **CONTEXT) is expected
    assert records(log)[0]["outcome"] == "build_error"


def wait_log(caplog, text, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(text in r.getMessage() for r in caplog.records):
            return
        time.sleep(0.01)
    raise AssertionError(f"no log record containing {text!r}")


def test_log_failure_never_raises_and_warns_once(fake_jeff, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    blocker = tmp_path / "file"
    blocker.write_text("not a dir")
    router = ShadowRouter(client=JeffClient(fake_jeff.url, 2.0), log=ShadowLog(blocker / "shadow.jsonl", 1000),
                          cwd=lambda: "", max_request_chars=24000)
    expected = main_text()
    for message in ("jeff_router: complete failed (", "jeff_router: complete failed again"):
        assert router(request=HI, next_call=NextCall(expected), **CONTEXT) is expected
        wait_log(caplog, message)  # the callback ran and swallowed the OSError
    warnings = [r for r in caplog.records
                if r.levelno == logging.WARNING and "jeff_router: complete failed" in r.getMessage()]
    assert len(warnings) == 1


def test_register_wires_middleware_and_log_path(tmp_path, fake_jeff):
    registered, hooks = [], []
    settings = {"url": fake_jeff.url}
    ctx = NS(get_config=lambda key, default: settings.get(key, default),
             register_middleware=lambda kind, fn: registered.append((kind, fn)),
             state=NS(data_dir=tmp_path / "plugin-data"), session_cwd=lambda: "", on_unload=hooks.append)
    register(ctx)
    [(kind, router)] = registered
    assert kind == "llm_execution"
    assert hooks == [router.close]
    router(request=HI, next_call=NextCall(main_text()), **CONTEXT)
    assert wait_records(tmp_path / "plugin-data" / "shadow.jsonl", 1)[0]["outcome"] == "ok"


def test_thinking_off_sends_reasoning_effort_none_to_the_main_model(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path, main_thinking=False)
    request = {"model": "m", **HI}
    before = json.dumps(request, sort_keys=True)
    nxt = NextCall(main_text())
    router(request=request, next_call=nxt, provider="local", **CONTEXT)
    [((sent,), _)] = nxt.calls
    assert sent == {**request, "reasoning_effort": "none"}
    assert json.dumps(request, sort_keys=True) == before
    assert wait_records(log, 1)[0]["main_thinking"] is False


def test_thinking_off_applies_to_any_chat_completions_provider(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path, main_thinking=False)
    nxt = NextCall(main_text())
    router(request=HI, next_call=nxt, provider="ollama-cloud", **CONTEXT)
    [((sent,), _)] = nxt.calls
    assert sent == {**HI, "reasoning_effort": "none"}
    assert wait_records(log, 1)[0]["main_thinking"] is False


def test_other_api_modes_pass_through_and_are_logged_once(fake_jeff, tmp_path, caplog):
    router, log = make_router(fake_jeff.url, tmp_path, main_thinking=False)
    context = {**CONTEXT, "api_mode": "anthropic_messages"}
    for _ in range(2):
        nxt = NextCall(main_text())
        router(request=HI, next_call=nxt, provider="anthropic", **context)
        assert nxt.calls == [((), {})]
    notes = [r for r in caplog.records if "api_mode anthropic_messages is not supported" in r.getMessage()]
    assert len(notes) == 1
    assert records(log) == []


def test_reasoning_despite_thinking_off_is_logged_once(fake_jeff, tmp_path, caplog):
    router, log = make_router(fake_jeff.url, tmp_path, main_thinking=False)
    reply = main_text()
    reply.choices[0].message.reasoning_content = "still thinking"
    for _ in range(2):
        router(request=HI, next_call=NextCall(reply), provider="ollama-cloud", **CONTEXT)
    wait_records(log, 2)
    notes = [r for r in caplog.records if "still returned 14 reasoning chars" in r.getMessage()]
    assert len(notes) == 1


def test_register_passes_main_thinking_setting(tmp_path, fake_jeff):
    registered = []
    settings = {"url": fake_jeff.url, "main_thinking": False}
    ctx = NS(get_config=lambda key, default: settings.get(key, default),
             register_middleware=lambda kind, fn: registered.append(fn), state=NS(data_dir=tmp_path))
    register(ctx)
    nxt = NextCall(main_text())
    registered[0](request=HI, next_call=nxt, provider="local", **CONTEXT)
    assert nxt.calls[0][0][0]["reasoning_effort"] == "none"


def test_register_without_state_registers_nothing_and_warns(caplog):
    caplog.set_level(logging.WARNING)
    registered = []
    # `hermes plugins validate` passes a stand-in that copies only callable PluginContext members (no `state`)
    ctx = NS(get_config=lambda key, default: default, register_middleware=lambda kind, fn: registered.append(kind))
    register(ctx)
    assert registered == []
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        "jeff_router: Hermes plugin context lacks state.data_dir; registering nothing"]


def test_register_disabled_registers_nothing(tmp_path):
    registered = []
    ctx = NS(get_config=lambda key, default: False if key == "enabled" else default,
             register_middleware=lambda kind, fn: registered.append(kind), state=NS(data_dir=tmp_path))
    register(ctx)
    assert registered == []


def test_close_stops_admission_and_passes_calls_through(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path)
    router.close()
    expected = main_text()
    assert router(request=HI, next_call=NextCall(expected), **CONTEXT) is expected
    time.sleep(0.1)
    assert fake_jeff.requests == [] and records(log) == []


def router_reply(p_xhigh, status=200):
    def reply(body):
        if body["model"] == "code-router":
            if status != 200:
                return status, b"err"
            return 200, ok_body({"q": {"off": 1 - p_xhigh, "low": 0.0, "medium": 0.0, "xhigh": p_xhigh}})
        return 200, ok_body({"tool": {"answer_directly": 1.0}})
    return reply


def make_routed(url, tmp_path):
    log_path = tmp_path / "shadow.jsonl"
    router = ShadowRouter(client=JeffClient(url, 2.0), log=ShadowLog(log_path, 10_000_000), cwd=lambda: "",
                          max_request_chars=24000, thinking_router=True)
    return router, log_path


def test_router_turns_thinking_off_below_threshold(fake_jeff, tmp_path):
    fake_jeff.reply = router_reply(0.3)
    router, log = make_routed(fake_jeff.url, tmp_path)
    request = {"model": "m", **HI}
    nxt = NextCall(main_text())
    router(request=request, next_call=nxt, provider="local", **CONTEXT)
    [((sent,), _)] = nxt.calls
    assert sent == {**request, "reasoning_effort": "none"}
    rec = wait_records(log, 1)[0]
    assert rec["main_thinking"] is False
    assert rec["router"]["level"] == "off" and rec["router"]["outcome"] == "ok"
    assert rec["router"]["probabilities"]["xhigh"] == 0.3
    assert fake_jeff.requests[0]["model"] == "code-router"  # asked before the call, ahead of the shadow question


def test_router_leaves_call_unchanged_at_or_above_threshold(fake_jeff, tmp_path):
    fake_jeff.reply = router_reply(0.6)
    router, log = make_routed(fake_jeff.url, tmp_path)
    nxt = NextCall(main_text())
    router(request=HI, next_call=nxt, provider="local", **CONTEXT)
    assert nxt.calls == [((), {})]
    rec = wait_records(log, 1)[0]
    assert rec["main_thinking"] is True and rec["router"]["level"] == "xhigh"


def test_router_failure_keeps_thinking(fake_jeff, tmp_path):
    fake_jeff.reply = router_reply(0.0, status=500)
    router, log = make_routed(fake_jeff.url, tmp_path)
    nxt = NextCall(main_text())
    router(request=HI, next_call=nxt, provider="local", **CONTEXT)
    assert nxt.calls == [((), {})]
    rec = wait_records(log, 1)[0]
    assert rec["router"] == {"level": "xhigh", "outcome": "server_error", "ms": rec["router"]["ms"],
                             "probabilities": None}


def test_router_asked_for_any_provider(fake_jeff, tmp_path):
    fake_jeff.reply = router_reply(0.0)
    router, log = make_routed(fake_jeff.url, tmp_path)
    nxt = NextCall(main_text())
    router(request=HI, next_call=nxt, provider="ollama-cloud", **CONTEXT)
    [((sent,), _)] = nxt.calls
    assert sent["reasoning_effort"] == "none"
    assert wait_records(log, 1)[0]["router"]["level"] == "off"
