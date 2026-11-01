import json
from pathlib import Path

from jeff_router_under_test import options
from jeff_router_under_test.options import (
    MAX_ALREADY_READ, MAX_PATH_PROBES, MAX_TOOLS, READ_INSTRUCTIONS, READ_ONLY_TOOLS, TOOL_INSTRUCTIONS,
    build_read_options, build_tool_options, normalize_path,
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


def test_already_read_accepts_dict_arguments(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("a")
    b.write_text("b")
    call = {"id": "c", "type": "function", "function": {"name": "read_file", "arguments": {"path": str(a)}}}
    messages = [msg("user", f"compare {a} and {b}"), msg("assistant", None, tool_calls=[call]),
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


def test_path_pattern_forms():
    for text in ("/x.md", "~/.bashrc", "~/notes.md", "./x.md", "../x.md", "docs/a.md", "/a/b.md"):
        assert options._PATH.search(text).group(0) == text
    for text in ("https://e.com/a", "word", "x.md"):
        assert options._PATH.search(text) is None


def test_home_relative_single_segment_path_is_a_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "notes.md"
    target.write_text("x")
    opts = build_read_options([msg("user", "summarize ~/notes.md please")], {"read_file"}, cwd="")
    assert opts.paths == {"r1": str(target.resolve())}


def test_normalize_path_accepts_missing_files(tmp_path):
    assert normalize_path("nope/x.txt", str(tmp_path)) == str((tmp_path / "nope" / "x.txt").resolve())
    assert normalize_path(None, str(tmp_path)) is None
    assert normalize_path("rel.txt", "") is None


def _count_probes(monkeypatch):
    calls = []
    real = options._existing_file

    def counting(raw, cwd):
        calls.append(raw)
        return real(raw, cwd)

    monkeypatch.setattr(options, "_existing_file", counting)
    return calls


def test_path_probes_are_bounded(tmp_path, monkeypatch):
    real = tmp_path / "real.txt"
    real.write_text("x")
    missing = " ".join(f"{tmp_path}/missing{i}.txt" for i in range(1000))
    calls = _count_probes(monkeypatch)
    assert build_read_options([msg("user", f"{missing} {real}")], {"read_file"}, cwd="") is None
    assert len(calls) == MAX_PATH_PROBES == 64
    calls.clear()
    opts = build_read_options([msg("user", f"{real} {missing}")], {"read_file"}, cwd="")
    assert opts.paths == {"r1": str(real.resolve())}
    assert len(calls) == MAX_PATH_PROBES


def test_repeated_raw_paths_are_probed_once(tmp_path, monkeypatch):
    real = tmp_path / "real.txt"
    real.write_text("x")
    calls = _count_probes(monkeypatch)
    text = " ".join([f"{tmp_path}/missing.txt"] * 500 + [str(real)])
    opts = build_read_options([msg("user", text), msg("assistant", text)], {"read_file"}, cwd="")
    assert opts.paths == {"r1": str(real.resolve())}
    assert len(calls) == 2


def test_already_read_scan_is_bounded_to_the_newest_calls(tmp_path, monkeypatch):
    call = lambda path: {"id": "c", "type": "function",  # noqa: E731
                         "function": {"name": "read_file", "arguments": json.dumps({"path": path})}}
    paths = [f"{tmp_path}/f{i}.txt" for i in range(1000)]
    messages = [msg("assistant", None, tool_calls=[call(p)]) for p in paths]
    calls = []
    real = options.normalize_path
    monkeypatch.setattr(options, "normalize_path", lambda raw, cwd: calls.append(raw) or real(raw, cwd))
    done = options._already_read(messages, cwd="")
    assert len(calls) == MAX_ALREADY_READ == 200
    assert done == {str(Path(p).resolve()) for p in paths[-MAX_ALREADY_READ:]}
