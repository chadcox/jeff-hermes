"""Jeff-Code's thinking-control guards, ported: runaway.ts (findRunaway) and loop-guard.ts (repeatedAction,
stuckOutputs, failedCommands), over a Hermes chat-completions message list. Jeff-Code's bash/write/edit are Hermes'
terminal/write_file/patch. Hermes has no scout, so every assistant message is the coding model's."""
from __future__ import annotations

import json
import re
from typing import Any

from .thinking import _output, _text

RUNAWAY_MIN_TEXT_CHARS = 20000
RUNAWAY_TAIL_CHARS = 400
RUNAWAY_MIN_PIECE_CHARS = 20
RUNAWAY_MIN_REPEATS = 5
RUNAWAY_LINE_WINDOW = 60
RUNAWAY_LINE_REPEATS = 8
RUNAWAY_MIN_LINE_CHARS = 20
RUNAWAY_CHECK_CHARS = 64

LOOKBACK = 6
SIMILARITY = 0.9
MIN_LENGTH_RATIO = 0.8
EXACT_BELOW_CHARS = 20
STUCK_OUTPUTS = 3
STUCK_TAIL_CHARS = 2000
FAILED_IN_A_ROW = 2

SHELL = "terminal"
WRITE = "write_file"
EDIT = "patch"

WRITING_PATTERNS = [re.compile(p) for p in (
    r"(?<![<>&=-])>>?(?![&=>])\s*(?!/dev/null\b)[^\s&|;<>()]",
    r"\btee\b",
    r"\b(?:sed|perl)\b[^|;&\n]*\s-[a-zA-Z]*i",
    r"(?:^|[\n;&|(]|\bsudo\b)\s*(?:cp|mv|rm|rmdir|touch|mkdir|ln|patch|truncate|dd|install|tar|unzip)\b",
    r"\bgit\s+(?:apply|am|checkout|restore|reset|stash|mv|rm|merge|rebase|cherry-pick|pull|clone)\b",
    r"\bopen\([^)]*,\s*['\"][wax]|\.write_(?:text|bytes)\(|\bshutil\.|\bos\.(?:remove|rename|replace|makedirs|mkdir|unlink)\b",
)]
CHANGES_FOLDER = re.compile(r"(?:^|[\s;&|(])(?:cd|pushd|popd)\b")
_PATH = r"""'[^'\n]+'|"[^"$`\\\n]+"|[^\s'"$`\\;&|<>()]+"""
_DELIMITER = r"""'\w+'|"\w+"|\w+"""
HEREDOC_WRITES = [re.compile(p) for p in (
    rf"^\s*cat\s+>\s*(?P<path>{_PATH})\s+<<(?P<dash>-?)\s*(?P<delimiter>{_DELIMITER})\s*$",
    rf"^\s*cat\s+<<(?P<dash>-?)\s*(?P<delimiter>{_DELIMITER})\s*>\s*(?P<path>{_PATH})\s*$",
)]


# ---- runaway.ts ----

def _repeated_piece(text: str) -> dict[str, Any] | None:
    tail = text[-RUNAWAY_TAIL_CHARS:]
    for period in range(RUNAWAY_MIN_PIECE_CHARS, RUNAWAY_TAIL_CHARS // RUNAWAY_MIN_REPEATS + 1):
        if all(tail[i] == tail[i - period] for i in range(period, len(tail))):
            return {"rule": "repeated_piece", "repeated": tail[-period:], "count": len(tail) // period}
    return None


def _repeated_line(text: str) -> dict[str, Any] | None:
    end = text.rfind("\n")  # the line still being written is not counted
    counts: dict[str, int] = {}
    for _ in range(RUNAWAY_LINE_WINDOW):
        if end < 0:
            break
        start = text.rfind("\n", 0, end)
        line = text[start + 1:end].strip()
        end = start
        if len(line) >= RUNAWAY_MIN_LINE_CHARS:
            counts[line] = counts.get(line, 0) + 1
    for line, count in counts.items():
        if count >= RUNAWAY_LINE_REPEATS:
            return {"rule": "repeated_line", "repeated": line, "count": count}
    return None


def find_runaway(text: str) -> dict[str, Any] | None:
    if len(text) < RUNAWAY_MIN_TEXT_CHARS:
        return None
    return _repeated_piece(text) or _repeated_line(text)


def first_runaway(text: str) -> dict[str, Any] | None:
    """Jeff-Code checks the streamed text every RUNAWAY_CHECK_CHARS; Hermes only hands over the finished reply, so
    replay those checks over its prefixes (a loop the model later broke out of still counts, as it would live)."""
    if len(text) < RUNAWAY_MIN_TEXT_CHARS:
        return None
    for end in range(RUNAWAY_MIN_TEXT_CHARS, len(text), RUNAWAY_CHECK_CHARS):
        hit = find_runaway(text[:end])
        if hit:
            return hit
    return find_runaway(text)


# ---- loop-guard.ts ----

def _normalised(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, list):
        return [_normalised(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalised(value[k]) for k in sorted(value)}
    return value


def _bigrams(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in range(len(text) - 1):
        counts[text[i:i + 2]] = counts.get(text[i:i + 2], 0) + 1
    return counts


def dice(a: str, b: str) -> float:
    if a == b:
        return 1.0
    ca, cb = _bigrams(a), _bigrams(b)
    total = sum(ca.values()) + sum(cb.values())
    if total == 0:
        return 0.0
    return 2 * sum(min(n, cb.get(k, 0)) for k, n in ca.items()) / total


def compare_texts(a: str, b: str) -> dict[str, Any]:
    shorter, longer = sorted((len(a), len(b)))
    similarity = dice(a, b)
    if a == b:
        same, rule = True, "equal"
    elif shorter < EXACT_BELOW_CHARS:
        same, rule = False, "short"
    elif shorter < MIN_LENGTH_RATIO * longer:
        same, rule = False, "length"
    else:
        same, rule = similarity >= SIMILARITY, "dice"
    return {"same": same, "rule": rule, "similarity": similarity, "lengths": [len(a), len(b)]}


def _call(call: Any) -> tuple[str, Any, str]:
    """(name, arguments, id) of an OpenAI-style tool call, dict or object."""
    get = (lambda o, k: o.get(k)) if isinstance(call, dict) else (lambda o, k: getattr(o, k, None))
    function = get(call, "function") or {}
    raw = get(function, "arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        args = raw
    return str(get(function, "name")), args if args is not None else {}, str(get(call, "id"))


def _command(args: Any) -> str | None:
    command = args.get("command") if isinstance(args, dict) else None
    return command if isinstance(command, str) else None


def _call_text(name: str, args: Any) -> str:
    command = _command(args)
    if name == SHELL and command is not None:
        return _normalised(command)
    return json.dumps(_normalised(args), separators=(",", ":"), ensure_ascii=False)


def writes_file(name: str, args: Any) -> bool:
    if name in (WRITE, EDIT):
        return True
    command = _command(args)
    if name != SHELL or command is None:
        return False
    return any(p.search(command) for p in WRITING_PATTERNS)


def compare_calls(next_call: tuple[str, Any], prev_call: tuple[str, Any]) -> dict[str, Any]:
    (name, next_args), (prev_name, prev_args) = next_call, prev_call
    if name != prev_name:
        return {"same": False, "rule": "name", "name": name}
    a, b = _call_text(name, next_args), _call_text(name, prev_args)
    if name != SHELL or writes_file(name, next_args) or writes_file(name, prev_args):
        rule = "equal" if a == b else ("arguments" if name != SHELL else "write")
        return {"same": a == b, "rule": rule, "name": name}
    return {**compare_texts(a, b), "name": name}


def _unquote(text: str) -> str:
    return text[1:-1] if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"" else text


def shell_file_writes(command: str) -> dict[str, Any]:
    lines = command.split("\n")
    writes: list[dict[str, str]] = []
    rest: list[str] = []
    i = 0
    while i < len(lines):
        match = next((m for m in (p.match(lines[i]) for p in HEREDOC_WRITES) if m), None)
        if match is None:
            rest.append(lines[i])
            i += 1
            continue
        raw = match["delimiter"]
        delimiter = _unquote(raw)
        strip = (lambda line: line.lstrip("\t")) if match["dash"] == "-" else (lambda line: line)
        end = next((j for j in range(i + 1, len(lines)) if strip(lines[j]) == delimiter), None)
        body = [strip(line) for line in lines[i + 1:end]] if end is not None else []
        expands = raw[0] not in "'\"" and any(c in line for line in body for c in "$`\\")
        if end is None or expands:
            rest.append(lines[i])
            i += 1
            continue
        writes.append({"path": _unquote(match["path"]), "content": "\n".join(body) + "\n" if body else ""})
        i = end + 1
    text = "\n".join(rest)
    unknown = any(p.search(text) for p in WRITING_PATTERNS) or (bool(writes) and bool(CHANGES_FOLDER.search(text)))
    return {"writes": writes, "unknown": unknown}


def _tool_results(messages: list[Any]) -> list[tuple[str, dict[str, Any]]]:
    """(tool name, result message) for each answered call, oldest first, each result once."""
    names = {}
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                name, _, call_id = _call(call)
                names[call_id] = name
    results, seen = [], set()
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "tool":
            call_id = str(message.get("tool_call_id"))
            if call_id in names and call_id not in seen:
                seen.add(call_id)
                results.append((names[call_id], message))
    return results


def write_effects(messages: list[Any]) -> dict[str, dict[str, Any]]:
    """Call id -> {changed, paths} for every past call that writes a file, in order."""
    errors = {str(m.get("tool_call_id")): _output(m.get("content"))[1] for _, m in _tool_results(messages)}
    known: dict[str, str] = {}
    effects: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            name, args, call_id = _call(call)
            if not writes_file(name, args):
                continue
            path = args.get("path") if isinstance(args, dict) else None
            if name in (WRITE, EDIT) and errors.get(call_id):
                effects[call_id] = {"changed": False, "paths": []}
            elif name == WRITE and isinstance(path, str) and isinstance(args.get("content"), str):
                effects[call_id] = {"changed": known.get(path) != args["content"], "paths": [path]}
                known[path] = args["content"]
            elif name in (WRITE, EDIT):
                if isinstance(path, str):
                    known.pop(path, None)
                effects[call_id] = {"changed": True, "paths": []}
            else:
                sw = shell_file_writes(_command(args) or "")
                if sw["unknown"]:
                    known.clear()
                changed = sw["unknown"]
                for write in sw["writes"]:
                    changed = changed or known.get(write["path"]) != write["content"]
                    known[write["path"]] = write["content"]
                effects[call_id] = {"changed": changed, "paths": [w["path"] for w in sw["writes"]]}
    return effects


def repeated_action(messages: list[Any], calls: list[Any]) -> dict[str, Any] | None:
    """The reply's calls repeat an earlier turn's within LOOKBACK turns, with no file changed in between."""
    if not calls:
        return None
    new = [_call(c)[:2] for c in calls]
    effects = write_effects(messages)
    turns_back = 0
    unchanged: list[dict[str, Any]] = []
    for message in reversed(messages):
        if turns_back >= LOOKBACK:
            break
        if not isinstance(message, dict) or message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        old = [_call(c) for c in message["tool_calls"]]
        turns_back += 1
        if len(old) == len(new):
            pairs = [compare_calls(n, o[:2]) for n, o in zip(new, old)]
            if all(p["same"] for p in pairs):
                return {"turns_back": turns_back, "similarity": min(p.get("similarity", 1.0) for p in pairs),
                        "pairs": pairs, "unchanged_writes": unchanged,
                        "repeated_action": [{"name": n, "arguments": a} for n, a in new]}
        for name, _, call_id in old:
            effect = effects.get(call_id)
            if effect is None:
                continue
            if effect["changed"]:
                return None
            unchanged.append({"name": name, "paths": effect["paths"], "turns_back": turns_back})
    return None


def stuck_outputs(messages: list[Any]) -> dict[str, Any] | None:
    results = _tool_results(messages)[-STUCK_OUTPUTS:]
    if len(results) < STUCK_OUTPUTS:
        return None
    tails = [_text(m.get("content"))[-STUCK_TAIL_CHARS:] for _, m in results]
    pairs = [compare_texts(tails[b], tails[a]) for a in range(len(tails)) for b in range(a + 1, len(tails))]
    return {"rule": "stuck_outputs", "pairs": pairs} if all(p["same"] for p in pairs) else None


def failed_commands(messages: list[Any]) -> dict[str, Any] | None:
    results = _tool_results(messages)[-FAILED_IN_A_ROW:]
    if len(results) < FAILED_IN_A_ROW:
        return None
    last_lines = []
    for name, message in results:
        shown, error = _output(message.get("content"))
        if name != SHELL or not error:
            return None
        last_lines.append(shown.rstrip().split("\n")[-1])
    return {"rule": "failed_commands", "last_lines": last_lines}


def previous_reply(messages: list[Any]) -> dict[str, Any] | None:
    """The last assistant reply in the history ran away, or repeated an earlier action with no file changed.

    Hermes can't re-ask a looped or runaway reply, so the next calls think instead. The check is made on the history
    each call, so it stops as soon as the latest reply is a new action."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "assistant":
            break
    else:
        return None
    for key in ("reasoning_content", "reasoning", "content"):
        text = message.get(key)
        hit = first_runaway(text) if isinstance(text, str) else None
        if hit:
            return {"rule": "previous_runaway", "where": key, "runaway": hit}
    repeat = repeated_action(messages[:index], list(message.get("tool_calls") or []))
    return {"rule": "previous_loop", **repeat} if repeat else None
