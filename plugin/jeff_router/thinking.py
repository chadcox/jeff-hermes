"""Jeff-Code's thinking router, ported: before each main-model call, Jeff's code-router adapter is asked how hard the
coding model should think, and the call runs with thinking off unless P(xhigh) >= threshold (Jeff-Code's
jeff-off-unless rule, packages/coding-agent/src/core/jeff-first/thinking.ts).

The question, options and state rendering are copied verbatim from Jeff-Code (router-question.ts, state.ts,
teacher-prompt.ts renderState) because the adapter was trained on exactly that text. Hermes's tool calls are shown as
the shell command that does the same, as Jeff-Code does for its own tools.
"""
from __future__ import annotations

import json
import shlex
import time
from typing import Any

from .jeff_client import JeffClient, JeffResult

ROUTER_QUESTION = "Before the coding model's next turn: how much should it think? Choose one option."
ROUTER_OPTIONS = {
    "off": "No thinking: it answers at once. Enough when the next step is obvious from what is shown.",
    "low": "Brief thinking before it answers.",
    "medium": "Some thinking before it answers.",
    "xhigh": "Long, careful thinking before it answers: it plans, checks its assumptions and considers other ways.",
}
STEP_BUDGET_CHARS = 8000
TERMINAL_LINES = 40
TERMINAL_LINE_CHARS = 200
BUSY_RETRIES = 3


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content
                         if isinstance(part, dict) and part.get("type") == "text")
    return ""


def shell_command(name: str, args: Any) -> str:
    """The shell command a Hermes tool call amounts to; otherwise the tool's name and its arguments."""
    a = args if isinstance(args, dict) else {}
    text = lambda key: a[key] if isinstance(a.get(key), str) else None  # noqa: E731
    whole = lambda key: a[key] if isinstance(a.get(key), int) and not isinstance(a.get(key), bool) else None  # noqa: E731
    path, command, pattern, content = text("path"), text("command"), text("pattern"), text("content")
    if name == "terminal" and command is not None:
        return command
    if name == "read_file" and path is not None:
        offset, limit = whole("offset"), whole("limit")
        if offset is None and limit is None:
            return f"cat {shlex.quote(path)}"
        first = offset or 1
        last = "$" if limit is None else str(first + limit - 1)
        return f"sed -n '{first},{last}p' {shlex.quote(path)}"
    if name == "search_files" and pattern is not None:
        where = shlex.quote(path or ".")
        if a.get("target") == "files":
            return f"find {where} -name {shlex.quote(pattern)}"
        return f"grep -rn {shlex.quote(pattern)} {where}"
    if name == "write_file" and path is not None and content is not None:
        return f"cat > {shlex.quote(path)} <<'EOF'\n{content}\nEOF"
    return f"{name} {json.dumps(a, ensure_ascii=False)}"


def _output(content: Any) -> tuple[str, bool]:
    """A Hermes tool result as the text a terminal would show, and whether it reported an error."""
    raw = _text(content)
    try:
        data = json.loads(raw)
    except ValueError:
        return raw, False
    if not isinstance(data, dict):
        return raw, False
    error = bool(data.get("error")) or data.get("success") is False or data.get("not_found") is True or (
        isinstance(data.get("exit_code"), int) and data["exit_code"] != 0)
    for key in ("output", "content", "matches_text", "diff"):
        if isinstance(data.get(key), str):
            shown = data[key]
            if error and isinstance(data.get("error"), str) and data["error"] not in shown:
                shown = f"{shown}\n{data['error']}" if shown else data["error"]
            return shown, error
    return raw, error


def _cut_lines(text: str, from_end: bool) -> str:
    lines = text.split("\n")
    if from_end:
        dropped = max(0, len(lines) - TERMINAL_LINES)
        shown = [line[:TERMINAL_LINE_CHARS] for line in lines[dropped:]]
        return "\n".join(([f"[{dropped} earlier lines not shown]"] if dropped else []) + shown)
    shown = [line[:TERMINAL_LINE_CHARS] for line in lines[:TERMINAL_LINES]]
    dropped = len(lines) - len(shown)
    return "\n".join(shown + ([f"[{dropped} more lines of this command not shown]"] if dropped else []))


def render_state(messages: list[Any]) -> str:
    """Jeff-Code's renderState(trimState(task, steps)) over a Hermes chat-completions message list."""
    users = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    task = _text(users[0].get("content")) if users else ""
    results = {m.get("tool_call_id"): m for m in messages if isinstance(m, dict) and m.get("role") == "tool"}
    steps: list[tuple[str, str | None, bool]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                args = function.get("arguments")
            result = results.get(call.get("id"))
            output, error = (None, False) if result is None else _output(result.get("content"))
            steps.append((shell_command(str(function.get("name")), args), output, error))
    kept: list[tuple[str, str | None, bool]] = []
    used = 0
    for command, output, error in reversed(steps):
        shown_output = None if output is None else _cut_lines(output, from_end=True)
        shown_command = _cut_lines(command, from_end=False)
        cost = len(shown_output or "") + len(shown_command)
        if used + cost > STEP_BUDGET_CHARS:
            break
        used += cost
        kept.append((shown_command, shown_output, error))
    kept.reverse()
    left_out = len(steps) - len(kept)
    parts = [f"Task:\n{task}"]
    if not kept:
        parts.append("No steps have been taken yet.")
    else:
        parts.append(f"Steps so far, oldest first ({left_out} earlier steps are not shown):" if left_out
                     else "Steps so far, oldest first:")
        for index, (command, output, error) in enumerate(kept, 1):
            note = "; the command reported an error" if error else ""
            parts.append(f"Step {index} (by the coding model{note}):\n$ {command}\n"
                         f"{output if output is not None else '(no output was recorded)'}")
    return "\n\n".join(parts)


def ask_level(client: JeffClient, adapter: str, state: str, threshold: float) -> tuple[str, JeffResult]:
    """"off" or "xhigh" by the jeff-off-unless rule; "xhigh" whenever Jeff gives no usable answer."""
    body = {"model": adapter, "state": state,
            "questions": {"q": {"type": "choice", "instructions": ROUTER_QUESTION, "criteria": ROUTER_OPTIONS}}}
    result = client.ask(body)
    for _ in range(BUSY_RETRIES):
        if result.outcome != "busy":
            break
        time.sleep(0.05)
        result = client.ask(body)
    if result.outcome != "ok":
        return "xhigh", result
    probabilities = (result.answers or {}).get("q", {}).get("probabilities", {})
    p_xhigh = probabilities.get("xhigh")
    if not isinstance(p_xhigh, (int, float)):
        return "xhigh", result
    return ("xhigh" if p_xhigh >= threshold else "off"), result
