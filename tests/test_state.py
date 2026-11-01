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
    assert len(fitted["user_message"]) >= 60


def long_tool_loop(entries):
    return {"agent": AGENT, "user_message": "u",
            "conversation": [{"role": "tool", "text": f"{i:04d}" + "x" * 8000} for i in range(entries)]}


def test_fit_state_trims_a_long_tool_loop_to_the_newest_entries():
    state = long_tool_loop(500)
    fitted = fit_state(state, 20_000)
    assert fitted is not None and size(fitted) <= 20_000
    assert fitted["conversation"][-1] == state["conversation"][-1]
    assert len(state["conversation"]) == 500  # input is not mutated


def test_fit_state_serializes_linearly(monkeypatch):
    state = long_tool_loop(500)
    total = size(state)
    calls = chars = 0
    dumps = json.dumps

    def counting_dumps(*args, **kwargs):
        nonlocal calls, chars
        out = dumps(*args, **kwargs)
        calls += 1
        chars += len(out)
        return out

    monkeypatch.setattr(json, "dumps", counting_dumps)
    fit_state(state, 20_000)
    assert calls < 3 * 500
    assert chars < 3 * total  # each entry is serialized a bounded number of times, not once per drop


def test_fit_state_gives_up_when_budget_is_tiny():
    assert fit_state({"agent": "a" * 100, "conversation": [], "user_message": "u"}, 20) is None
