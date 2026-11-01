"""Questions for Jeff: the tool list (tools adapter format) and code-built read_file candidates."""
from __future__ import annotations

import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Copy of Hermes' NO_EFFECT_TOOL_NAMES (agent/tool_result_classification.py:16-20); not imported on purpose.
READ_ONLY_TOOLS = frozenset({
    "read_file", "search_files", "session_search", "skill_view", "skills_list",
    "web_extract", "web_search", "vision_analyze", "browser_snapshot",
    "browser_get_images", "browser_console", "read_terminal",
})
TOOL_INSTRUCTIONS = (
    "Which tool should the agent call next to handle the user's latest message? Use the conversation for context. "
    "If no tool is needed, choose answer directly. If a tool is needed but information it requires is missing, "
    "choose ask the user."
)
ANSWER_DIRECTLY = "No tool is needed: answer the user directly from the conversation"
ASK_USER = "A tool is needed but required information is missing: ask the user first"
READ_INSTRUCTIONS = (
    "Which file should the agent read next to handle the user's latest message? If none of these files is the "
    "right one, choose none of these."
)
NONE_OF_THESE = "None of these: the agent needs a different file or a different tool"
MAX_TOOLS = 150
MAX_READ_CANDIDATES = 10
MAX_READ_BYTES = 2 * 1024 * 1024
DESCRIPTION_CHARS = 120
SCAN_MESSAGES = 12
MAX_PATH_PROBES = 64  # filesystem stats per question; runs on the agent thread before the model call
MAX_ALREADY_READ = 200  # newest read_file calls checked for "already read"; each one resolves a path
# Forms starting with /, ~/, ./ or ../ may have one segment; bare relative paths need a slash. Not inside URLs.
_PATH = re.compile(r"(?<![\w:/.~-])(?:(?:~|\.{1,2})?/[A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)*"
                   r"|[A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)+)")


@dataclass(frozen=True)
class ToolOptions:
    criteria: dict[str, str]
    names: dict[str, str]
    truncated: bool


@dataclass(frozen=True)
class ReadOptions:
    criteria: dict[str, str]
    paths: dict[str, str]


def _first_sentence(text: str) -> str:
    text = " ".join(str(text).split())
    match = re.search(r"(?<=[.!?])\s", text)
    sentence = text[:match.start()] if match else text
    if len(sentence) > DESCRIPTION_CHARS:
        sentence = sentence[:DESCRIPTION_CHARS - 1].rstrip() + "…"
    return sentence


def _signature(tool: Any) -> tuple[str, str] | None:
    function = tool.get("function") if isinstance(tool, dict) else None
    name = function.get("name") if isinstance(function, dict) else None
    if not isinstance(name, str) or not name:
        return None
    parameters = function.get("parameters")
    required = parameters.get("required") if isinstance(parameters, dict) else None
    args = ", ".join(str(r) for r in required) if isinstance(required, list) else ""
    description = _first_sentence(function.get("description") or "")
    return name, f"{name}({args}): {description}" if description else f"{name}({args})"


def build_tool_options(tools: list[Any]) -> ToolOptions:
    signatures: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tool in tools if isinstance(tools, list) else []:
        signature = _signature(tool)
        if signature and signature[0] not in seen:
            seen.add(signature[0])
            signatures.append(signature)
    truncated = len(signatures) > MAX_TOOLS
    if truncated:
        ranked = ([s for s in signatures if s[0] in READ_ONLY_TOOLS]
                  + [s for s in signatures if s[0] not in READ_ONLY_TOOLS])
        keep = {name for name, _ in ranked[:MAX_TOOLS]}
        signatures = [s for s in signatures if s[0] in keep]
    criteria = {"answer_directly": ANSWER_DIRECTLY, "ask_user": ASK_USER}
    names: dict[str, str] = {}
    for index, (name, text) in enumerate(signatures, 1):
        criteria[f"t{index}"] = text
        names[f"t{index}"] = name
    return ToolOptions(criteria, names, truncated)


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part["text"] for part in content
                         if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


def normalize_path(raw: Any, cwd: str) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        path = Path(raw.strip()).expanduser()
        if not path.is_absolute():
            if not cwd:
                return None
            path = Path(cwd) / path
        return str(path.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return None


def _existing_file(raw: str, cwd: str) -> str | None:
    normalized = normalize_path(raw, cwd)
    if normalized is None:
        return None
    try:
        info = Path(normalized).stat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_READ_BYTES:
        return None
    return normalized


def _already_read(messages: list[Any], cwd: str) -> set[str]:
    """Paths read whole by the newest MAX_ALREADY_READ read_file calls."""
    read: set[str] = set()
    seen = 0
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in reversed(message.get("tool_calls") or []):
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or function.get("name") != "read_file":
                continue
            if seen == MAX_ALREADY_READ:
                return read
            seen += 1
            raw = function.get("arguments")
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
            except (TypeError, ValueError):
                continue
            if isinstance(args, dict) and "offset" not in args and "limit" not in args:
                normalized = normalize_path(args.get("path"), cwd)
                if normalized:
                    read.add(normalized)
    return read


def _texts_newest_first(messages: list[Any]) -> list[str]:
    convo = [m for m in messages if isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool")]
    last_user = next((m for m in reversed(convo) if m.get("role") == "user"), None)
    texts = [message_text(last_user.get("content"))] if last_user else []
    texts += [message_text(m.get("content")) for m in reversed(convo[-SCAN_MESSAGES:]) if m is not last_user]
    return texts


def build_read_options(messages: list[Any], tool_names: set[str], cwd: str) -> ReadOptions | None:
    if "read_file" not in tool_names or not isinstance(messages, list):
        return None
    done = _already_read(messages, cwd)
    found: list[str] = []
    probed: set[str] = set()
    for raw in (m.group(0).rstrip(".,:;") for text in _texts_newest_first(messages) for m in _PATH.finditer(text)):
        if raw in probed:
            continue
        probed.add(raw)
        path = _existing_file(raw, cwd)
        if path and path not in done and path not in found:
            found.append(path)
        if len(found) == MAX_READ_CANDIDATES or len(probed) == MAX_PATH_PROBES:
            break
    if not found:
        return None
    criteria: dict[str, str] = {}
    paths: dict[str, str] = {}
    for index, path in enumerate(found, 1):
        key = f"r{index}"
        criteria[key] = f'read_file(path="{path}"): read {Path(path).name}'
        paths[key] = path
    criteria["none_of_these"] = NONE_OF_THESE
    return ReadOptions(criteria, paths)
