import json
from types import SimpleNamespace as NS

from jeff_router_under_test.guard import (
    compare_texts, dice, failed_commands, find_runaway, first_runaway, previous_reply, repeated_action,
    shell_file_writes,
    stuck_outputs,
)
from test_middleware import CONTEXT, HI, NextCall, make_router, make_routed, router_reply, main_text, wait_records


def call(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def step(call_id, name, args, output="ok", exit_code=0):
    return [{"role": "assistant", "content": None, "tool_calls": [call(call_id, name, args)]},
            {"role": "tool", "tool_call_id": call_id,
             "content": json.dumps({"output": output, "exit_code": exit_code})}]


def history(*steps):
    messages = [{"role": "user", "content": "task"}]
    for s in steps:
        messages += s
    return messages


# runaway

def test_short_text_is_never_a_runaway():
    assert find_runaway("ab" * 5000) is None


def test_repeated_piece_reports_shortest_period():
    piece = "the same thought again. "  # 24 chars
    hit = find_runaway("x" * 20000 + piece * 40)
    assert hit == {"rule": "repeated_piece", "repeated": piece, "count": 400 // 24}


def test_repeated_line_counts_complete_lines_only():
    lines = "".join(f"distinct line number {i:05d}\n" for i in range(1000))
    text = lines + "I will check the file again now.\n" * 8 + "partial"
    hit = find_runaway(text)
    assert hit is not None and hit["rule"] in ("repeated_line", "repeated_piece")
    assert find_runaway(lines + "partial") is None


def test_first_runaway_finds_a_runaway_that_later_text_hides():
    piece = "loop loop loop loop! "
    text = "y" * 20000 + piece * 30 + "".join(f"fresh unique line {i}\n" for i in range(200))
    assert find_runaway(text) is None
    assert first_runaway(text) is not None


# loop guard primitives

def test_dice_and_compare_texts():
    assert dice("abc", "abc") == 1
    assert compare_texts("same", "same")["rule"] == "equal"
    assert compare_texts("short", "other")["rule"] == "short"
    assert compare_texts("a" * 30, "a" * 100)["rule"] == "length"
    assert compare_texts("pytest tests/test_a.py -q", "pytest tests/test_a.py -x")["same"] is True


def test_shell_file_writes_heredoc_and_redirect():
    assert shell_file_writes("cat > a.py <<'EOF'\nprint(1)\nEOF") == {
        "writes": [{"path": "a.py", "content": "print(1)\n"}], "unknown": False}
    assert shell_file_writes("echo hi > out.txt")["unknown"] is True
    assert shell_file_writes("ls -la")["unknown"] is False


def test_stuck_outputs_needs_three_same_tails():
    same = [step(f"c{i}", "terminal", {"command": f"python run{i}.py"}, "Traceback: boom") for i in range(3)]
    assert stuck_outputs(history(*same))["rule"] == "stuck_outputs"
    assert stuck_outputs(history(*same[:2])) is None
    mixed = same[:2] + [step("c9", "terminal", {"command": "ls"}, "a.py b.py c.py d.py e.py")]
    assert stuck_outputs(history(*mixed)) is None


def test_failed_commands_needs_two_failed_terminal_calls():
    fails = [step(f"c{i}", "terminal", {"command": "make"}, f"line one\nerror {i}", 2) for i in range(2)]
    assert failed_commands(history(*fails)) == {"rule": "failed_commands", "last_lines": ["error 0", "error 1"]}
    ok = fails[:1] + [step("c5", "terminal", {"command": "make"}, "built")]
    assert failed_commands(history(*ok)) is None


def test_repeated_action_spots_a_repeat_with_no_change_between():
    messages = history(step("c1", "read_file", {"path": "a.py"}), step("c2", "terminal", {"command": "ls"}))
    hit = repeated_action(messages, [call("n1", "read_file", {"path": "a.py"})])
    assert hit is not None and hit["turns_back"] == 2 and hit["similarity"] == 1


def test_repeated_action_is_cleared_by_a_changing_write():
    messages = history(step("c1", "read_file", {"path": "a.py"}),
                       step("c2", "write_file", {"path": "a.py", "content": "new"}))
    assert repeated_action(messages, [call("n1", "read_file", {"path": "a.py"})]) is None


def test_repeated_action_ignores_new_calls():
    messages = history(step("c1", "read_file", {"path": "a.py"}))
    assert repeated_action(messages, [call("n1", "read_file", {"path": "b.py"})]) is None


# middleware wiring

def reply_calls(*calls):
    tool_calls = [NS(id=c["id"], type="function", function=NS(**c["function"])) for c in calls]
    return NS(choices=[NS(index=0, finish_reason="tool_calls",
                          message=NS(role="assistant", content=None, tool_calls=tool_calls))])


def reply_text(text):
    return NS(choices=[NS(index=0, finish_reason="stop", message=NS(role="assistant", content=text,
                                                                  tool_calls=None))])


def test_guard_forced_call_skips_jeff(fake_jeff, tmp_path):
    fake_jeff.reply = router_reply(0.0)
    router, log = make_routed(fake_jeff.url, tmp_path)
    fails = [step(f"c{i}", "terminal", {"command": "make"}, "error", 1) for i in range(2)]
    request = {"messages": history(*fails), "tools": HI["tools"]}
    nxt = NextCall(main_text())
    router(request=request, next_call=nxt, provider="local", **CONTEXT)
    assert nxt.calls == [((), {})]
    rec = wait_records(log, 1)[0]
    assert rec["main_thinking"] is True and rec["guard"]["forced"][0]["rule"] == "failed_commands"
    assert rec["router"] is None and router_asked(fake_jeff) == []  # the guard decided, so Jeff isn't asked


def test_loop_with_thinking_off_is_logged_not_reasked(fake_jeff, tmp_path):
    fake_jeff.reply = router_reply(0.0)
    router, log = make_routed(fake_jeff.url, tmp_path)
    request = {"messages": history(step("c1", "read_file", {"path": "a.py"})), "tools": HI["tools"]}
    first = reply_calls(call("n1", "read_file", {"path": "a.py"}))
    nxt = NextCall(first)
    assert router(request=request, next_call=nxt, provider="local", **CONTEXT) is first
    assert [c[0] for c in nxt.calls] == [({**request, "reasoning_effort": "none"},)]
    rec = wait_records(log, 1)[0]
    assert rec["main_thinking"] is False and rec["guard"]["trigger"]["trigger"] == "loop"


def test_runaway_is_logged_not_reasked(fake_jeff, tmp_path):
    fake_jeff.reply = router_reply(1.0)
    router, log = make_routed(fake_jeff.url, tmp_path)
    first = reply_text("z" * 20000 + "round and round we go " * 40)
    nxt = NextCall(first)
    assert router(request=HI, next_call=nxt, provider="local", **CONTEXT) is first
    assert len(nxt.calls) == 1
    rec = wait_records(log, 1)[0]
    assert rec["guard"]["trigger"]["trigger"] == "runaway" and rec["guard"]["trigger"]["where"] == "text"


def test_no_trigger_means_one_call_and_no_guard(fake_jeff, tmp_path):
    fake_jeff.reply = router_reply(0.0)
    router, log = make_routed(fake_jeff.url, tmp_path)
    nxt = NextCall(main_text())
    router(request=HI, next_call=nxt, provider="local", **CONTEXT)
    assert len(nxt.calls) == 1
    assert wait_records(log, 1)[0]["guard"] is None


# previous reply (recovery after a loop or runaway that could not be re-asked)

def test_previous_reply_spots_a_loop_and_clears_on_a_new_action():
    looped = history(step("c1", "read_file", {"path": "a.py"}), step("c2", "read_file", {"path": "a.py"}))
    assert previous_reply(looped)["rule"] == "previous_loop"
    moved_on = looped + step("c3", "terminal", {"command": "pytest"})
    assert previous_reply(moved_on) is None
    assert previous_reply(history()) is None


def test_previous_reply_spots_a_runaway():
    messages = history() + [{"role": "assistant", "content": "z" * 20000 + "round and round we go " * 40}]
    assert previous_reply(messages)["rule"] == "previous_runaway"


# guarded-off: main_thinking false, guards true

def router_asked(fake_jeff):
    return [r for r in fake_jeff.requests if r.get("model") == "code-router"]


def test_guarded_off_sends_effort_none_and_never_asks_jeff(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path, main_thinking=False)
    nxt = NextCall(main_text())
    router(request=HI, next_call=nxt, provider="local", **CONTEXT)
    assert nxt.calls == [(({**HI, "reasoning_effort": "none"},), {})]
    rec = wait_records(log, 1)[0]
    assert rec["main_thinking"] is False and rec["guard"] is None and rec["router"] is None
    assert router_asked(fake_jeff) == []


def test_guarded_off_thinks_on_failed_commands(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path, main_thinking=False)
    fails = [step(f"c{i}", "terminal", {"command": "make"}, "error", 1) for i in range(2)]
    request = {"messages": history(*fails), "tools": HI["tools"]}
    nxt = NextCall(main_text())
    router(request=request, next_call=nxt, provider="local", **CONTEXT)
    assert nxt.calls == [((), {})]
    rec = wait_records(log, 1)[0]
    assert rec["main_thinking"] is True and rec["guard"]["forced"][0]["rule"] == "failed_commands"
    assert router_asked(fake_jeff) == []


def test_guarded_off_thinks_on_stuck_outputs(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path, main_thinking=False)
    same = [step(f"c{i}", "terminal", {"command": f"python run{i}.py"}, "Traceback: boom") for i in range(3)]
    nxt = NextCall(main_text())
    router(request={"messages": history(*same), "tools": HI["tools"]}, next_call=nxt, provider="local", **CONTEXT)
    assert nxt.calls == [((), {})]
    assert wait_records(log, 1)[0]["guard"]["forced"][0]["rule"] == "stuck_outputs"


def test_guarded_off_thinks_after_a_logged_loop_until_the_reply_changes(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path, main_thinking=False)
    looped = history(step("c1", "read_file", {"path": "a.py"}), step("c2", "read_file", {"path": "a.py"}))
    nxt = NextCall(main_text())
    router(request={"messages": looped, "tools": HI["tools"]}, next_call=nxt, provider="local", **CONTEXT)
    assert nxt.calls == [((), {})]
    assert wait_records(log, 1)[0]["guard"]["forced"][0]["rule"] == "previous_loop"
    moved_on = looped + step("c3", "terminal", {"command": "pytest"}, "3 passed in 0.12s")
    nxt = NextCall(main_text())
    router(request={"messages": moved_on, "tools": HI["tools"]}, next_call=nxt, provider="local", **CONTEXT)
    assert nxt.calls[0][0][0]["reasoning_effort"] == "none"


def test_guards_false_never_forces_thinking_or_logs_triggers(fake_jeff, tmp_path):
    router, log = make_router(fake_jeff.url, tmp_path, main_thinking=False, guards=False)
    fails = [step(f"c{i}", "terminal", {"command": "make"}, "error", 1) for i in range(2)]
    request = {"messages": history(*fails), "tools": HI["tools"]}
    nxt = NextCall(reply_text("z" * 20000 + "round and round we go " * 40))
    router(request=request, next_call=nxt, provider="local", **CONTEXT)
    assert nxt.calls[0][0][0]["reasoning_effort"] == "none"
    rec = wait_records(log, 1)[0]
    assert rec["main_thinking"] is False and rec["guard"] is None
