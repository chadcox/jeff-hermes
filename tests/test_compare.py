import json
from types import SimpleNamespace as NS

from jeff_router_under_test.compare import (
    MAX_ARGUMENT_CHARS, logged_action, pick, read_match, main_action, tool_match,
)


def response(content=None, calls=None):
    tool_calls = [NS(id=f"c{i}", type="function", function=NS(name=n, arguments=a)) for i, (n, a) in
                  enumerate(calls or [])] or None
    return NS(choices=[NS(index=0, finish_reason="stop", message=NS(role="assistant", content=content,
                                                                    tool_calls=tool_calls))])


def test_text_response():
    assert main_action(response(content="hello")) == {"kind": "text"}


def test_tool_response_first_call_parsed():
    action = main_action(response(calls=[("read_file", json.dumps({"path": "/a"})), ("terminal", "{}")]))
    assert action == {"kind": "tool", "tools": ["read_file", "terminal"],
                      "first": {"name": "read_file", "arguments": {"path": "/a"}}}


def test_dict_shaped_response_and_bad_arguments():
    raw = {"choices": [{"message": {"tool_calls": [{"function": {"name": "x", "arguments": "not json"}}]}}]}
    assert main_action(raw) == {"kind": "tool", "tools": ["x"], "first": {"name": "x",
                                                                            "arguments": {"_raw": "not json"}}}


def test_comparisons_see_uncapped_arguments_and_only_the_logged_copy_is_capped():
    big = "x" * 5000
    arguments = {"path": "/a", "content": big, "edits": [{"old": big, "n": 3}], "flag": True}
    action = main_action(response(calls=[("write_file", json.dumps(arguments))]))
    assert action["first"]["arguments"] == arguments
    capped = "x" * MAX_ARGUMENT_CHARS + "…"
    assert MAX_ARGUMENT_CHARS == 2000
    logged = logged_action(action)
    assert logged["first"]["arguments"] == {"path": "/a", "content": capped, "edits": [{"old": capped, "n": 3}],
                                            "flag": True}
    assert logged["kind"] == "tool" and logged["tools"] == ["write_file"] and logged["first"]["name"] == "write_file"
    assert action["first"]["arguments"]["content"] == big  # the logged copy does not change the action
    raw = {"choices": [{"message": {"tool_calls": [{"function": {"name": "w", "arguments": arguments}}]}}]}
    assert logged_action(main_action(raw))["first"]["arguments"]["content"] == capped
    assert arguments["content"] == big  # dict-form arguments are not changed in place
    assert logged_action({"kind": "text"}) == {"kind": "text"} and logged_action(None) is None


def test_read_match_on_a_path_longer_than_the_log_cap():
    path = "/" + "p" * 2499
    action = main_action(response(calls=[("read_file", json.dumps({"path": path}))]))
    assert read_match("r1", {"r1": path}, action, "/") is True


def test_missing_choices():
    assert main_action(NS(choices=[])) is None
    assert main_action(None) is None


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
    nameless = {"kind": "tool", "tools": [None], "first": {"name": None, "arguments": {}}}
    assert tool_match("t1", names, nameless) is None
    assert tool_match("t9", names, tool) is None
    assert tool_match("answer_directly", names, nameless) is False


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
    no_args = {"kind": "tool", "tools": ["read_file"], "first": {"name": "read_file", "arguments": {}}}
    assert read_match("none_of_these", paths, no_args, cwd="") is None
    assert read_match("r1", paths, read("a.md"), cwd="") is None
