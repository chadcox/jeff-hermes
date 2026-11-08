import time
import json

from conftest import ok_body
from jeff_router_under_test.thinking import ROUTER_OPTIONS, ROUTER_QUESTION, ask_level, render_state, shell_command
from jeff_router_under_test.jeff_client import JeffClient


def call(id_, name, **args):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_question_and_options_match_jeff_code_verbatim():
    assert ROUTER_QUESTION == "Before the coding model's next turn: how much should it think? Choose one option."
    assert list(ROUTER_OPTIONS) == ["off", "low", "medium", "xhigh"]
    assert ROUTER_OPTIONS["off"].startswith("No thinking: it answers at once.")


def test_hermes_tools_render_as_shell_commands():
    assert shell_command("terminal", {"command": "ls -la"}) == "ls -la"
    assert shell_command("read_file", {"path": "a b.py"}) == "cat 'a b.py'"
    assert shell_command("read_file", {"path": "x.py", "offset": 10, "limit": 5}) == "sed -n '10,14p' x.py"
    assert shell_command("read_file", {"path": "x.py", "offset": 10}) == "sed -n '10,$p' x.py"
    assert shell_command("search_files", {"pattern": "def f"}) == "grep -rn 'def f' ."
    assert shell_command("search_files", {"pattern": "*.py", "target": "files", "path": "src"}) == "find src -name '*.py'"
    assert shell_command("write_file", {"path": "f", "content": "x"}) == "cat > f <<'EOF'\nx\nEOF"
    assert shell_command("patch", {"path": "f"}) == 'patch {"path": "f"}'


def test_render_state_matches_jeff_code_layout():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the test"},
        {"role": "assistant", "content": None, "tool_calls": [call("a", "terminal", command="pytest")]},
        {"role": "tool", "tool_call_id": "a", "content": json.dumps({"output": "1 failed", "exit_code": 1,
                                                                      "error": None})},
        {"role": "assistant", "content": None, "tool_calls": [call("b", "read_file", path="t.py")]},
        {"role": "tool", "tool_call_id": "b", "content": json.dumps({"content": "     1|x = 1"})},
        {"role": "assistant", "content": None, "tool_calls": [call("c", "terminal", command="make")]},
    ]
    assert render_state(messages) == (
        "Task:\nfix the test\n\nSteps so far, oldest first:\n\n"
        "Step 1 (by the coding model; the command reported an error):\n$ pytest\n1 failed\n\n"
        "Step 2 (by the coding model):\n$ cat t.py\n     1|x = 1\n\n"
        "Step 3 (by the coding model):\n$ make\n(no output was recorded)")


def test_render_state_without_steps():
    assert render_state([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]) == (
        "Task:\nhi\n\nNo steps have been taken yet.")


def test_render_state_keeps_user_updates_and_tool_history():
    messages = [
        {"role": "user", "content": "say hi"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "Analyze the concurrency bug"},
        {"role": "assistant", "tool_calls": [call("a", "terminal", command="pytest")]},
        {"role": "tool", "tool_call_id": "a", "content": "race detected"},
        {"role": "user", "content": [{"type": "text", "text": "Preserve the public API"}]},
    ]
    state = render_state(messages)
    assert state.startswith("Task:\nsay hi\n\nUser update:\nAnalyze the concurrency bug\n\n"
                            "User update:\nPreserve the public API\n\nSteps so far, oldest first:")
    assert "$ pytest\nrace detected" in state


def test_render_state_keeps_terminal_tail_and_budget():
    long_output = "\n".join(f"line {i}" for i in range(100))
    messages: list[dict] = [{"role": "user", "content": "t"}]
    for i in range(40):
        messages.append({"role": "assistant", "tool_calls": [call(str(i), "terminal", command=f"cmd{i}")]})
        messages.append({"role": "tool", "tool_call_id": str(i), "content": long_output})
    state = render_state(messages)
    assert "[60 earlier lines not shown]" in state
    assert "earlier steps are not shown):" in state
    assert "$ cmd39\n" in state and "$ cmd0\n" not in state


def test_ask_level_off_unless_xhigh_reaches_threshold(fake_jeff):
    client = JeffClient(fake_jeff.url, 2.0)
    fake_jeff.reply = (200, ok_body({"q": {"off": 0.5, "low": 0.0, "medium": 0.0, "xhigh": 0.5}}))
    assert ask_level(client, "code-router", "s", 0.6)[0] == "off"
    fake_jeff.reply = (200, ok_body({"q": {"off": 0.3, "low": 0.0, "medium": 0.0, "xhigh": 0.6}}))
    assert ask_level(client, "code-router", "s", 0.6)[0] == "xhigh"
    sent = fake_jeff.requests[-1]
    assert sent["model"] == "code-router" and sent["state"] == "s"
    assert sent["questions"] == {"q": {"type": "choice", "instructions": ROUTER_QUESTION, "criteria": ROUTER_OPTIONS}}


def test_ask_level_thinks_when_jeff_fails_and_retries_busy(fake_jeff):
    client = JeffClient(fake_jeff.url, 2.0)
    fake_jeff.reply = (500, b"boom")
    level, result = ask_level(client, "code-router", "s", 0.6)
    assert (level, result.outcome) == ("xhigh", "server_error")
    replies = [(529, b"busy"), (200, ok_body({"q": {"off": 0.9, "xhigh": 0.1}}))]
    fake_jeff.reply = lambda body: replies.pop(0)
    assert ask_level(client, "code-router", "s", 0.6)[0] == "off"


def test_ask_level_nan_xhigh_keeps_thinking(fake_jeff):
    client = JeffClient(fake_jeff.url, 2.0)
    fake_jeff.reply = (200, ok_body({"q": {"off": 0.9, "xhigh": float("nan")}}))
    level, result = ask_level(client, "code-router", "s", 0.6)
    assert (level, result.outcome) == ("xhigh", "bad_response")


def test_ask_level_missing_xhigh_is_bad_response(fake_jeff):
    client = JeffClient(fake_jeff.url, 2.0)
    fake_jeff.reply = (200, ok_body({"q": {"off": 0.9, "low": 0.1}}))
    level, result = ask_level(client, "code-router", "s", 0.6)
    assert (level, result.outcome) == ("xhigh", "bad_response")


def test_ask_level_logs_total_time_and_attempts_across_busy_retries(fake_jeff):
    replies = [(529, b"busy"), (200, ok_body({"q": {"off": 0.9, "low": 0.0, "medium": 0.0, "xhigh": 0.1}}))]
    fake_jeff.reply = lambda body: replies.pop(0)
    started = time.monotonic()
    level, result = ask_level(JeffClient(fake_jeff.url, 2.0), "code-router", "s", 0.6)
    elapsed_ms = (time.monotonic() - started) * 1000
    assert level == "off" and result.attempts == 2
    assert result.ms >= 50 and result.ms <= elapsed_ms + 1  # includes the 50 ms wait between attempts


# 3.1 budgeting the whole router input

def size(text):
    return len(json.dumps(text, ensure_ascii=False)) - 2


def terminal(call_id, command, output):
    return [{"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": "terminal", "arguments": json.dumps({"command": command})}}]},
            {"role": "tool", "tool_call_id": call_id, "content": json.dumps({"output": output, "exit_code": 0})}]


def test_render_state_without_budget_is_unchanged_for_small_input():
    messages = [{"role": "user", "content": "fix it"}] + terminal("c1", "ls", "a.py")
    assert render_state(messages) == render_state(messages, 24000)


def test_long_first_task_stays_in_budget_and_keeps_the_newest_step():
    messages = [{"role": "user", "content": "do this " * 20000}] + terminal("c1", "pytest", "3 passed")
    state = render_state(messages, 20000)
    assert size(state) <= 20000 and state.startswith("Task:\ndo this") and "[...]" in state
    assert "$ pytest\n3 passed" in state


def test_long_follow_ups_keep_the_current_instruction():
    messages = ([{"role": "user", "content": "build the app"}]
                + [{"role": "user", "content": "x" * 30000} for _ in range(3)]
                + [{"role": "user", "content": "now add tests"}] + terminal("c1", "ls", "a.py"))
    state = render_state(messages, 20000)
    assert size(state) <= 20000
    assert "build the app" in state and "now add tests" in state and "[3 earlier user updates not shown]" in state


def test_forty_max_width_lines_are_cut_not_dropped():
    output = "\n".join("y" * 200 for _ in range(40))
    messages = [{"role": "user", "content": "go"}] + terminal("c1", "ls", "a") + terminal("c2", "make", output)
    state = render_state(messages)
    assert "No steps have been taken yet." not in state and "$ make" in state and "[...]" in state
    budgeted = render_state(messages, 6000)
    assert size(budgeted) <= 6000 and "$ make" in budgeted


def test_too_small_a_budget_returns_none():
    messages = [{"role": "user", "content": "go"}] + terminal("c1", "make", "z" * 5000)
    assert render_state(messages, 10) is None
