"""The tools adapter's state (agent, conversation, user_message) built from Hermes request messages."""
from __future__ import annotations

import json
from typing import Any

from .options import message_text

AGENT = "A personal assistant with shell, file, web, memory and other tools on the user's Linux workstation."
MAX_CONVERSATION = 6
TOOL_LINES = 40
LINE_CHARS = 200
CALL_CHARS = 300
ROLES = ("user", "assistant", "tool")


def _tail(text: str) -> str:
    return "\n".join(line[:LINE_CHARS] for line in text.splitlines()[-TOOL_LINES:])


def _render(message: dict[str, Any]) -> dict[str, str]:
    role = message["role"]
    if role == "tool":
        return {"role": role, "text": _tail(message_text(message.get("content")))}
    parts = [message_text(message.get("content")).strip()]
    if role == "assistant":
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict):
                parts.append(f"-> {function.get('name', '?')}({function.get('arguments', '')})"[:CALL_CHARS])
    return {"role": role, "text": "\n".join(p for p in parts if p)}


def build_state(messages: list[Any]) -> dict[str, Any]:
    convo = [m for m in messages if isinstance(m, dict) and m.get("role") in ROLES] if isinstance(messages, list) else []
    last_user = max((i for i, m in enumerate(convo) if m["role"] == "user"), default=None)
    if last_user is None:
        before, after, user_message = convo[-MAX_CONVERSATION:], [], ""
    else:
        before, after = convo[:last_user][-MAX_CONVERSATION:], convo[last_user + 1:]
        user_message = message_text(convo[last_user].get("content"))
    return {"agent": AGENT, "conversation": [_render(m) for m in before + after], "user_message": user_message}


def _size(state: dict[str, Any]) -> int:
    return len(json.dumps(state, ensure_ascii=False))


def fit_state(state: dict[str, Any], budget: int) -> dict[str, Any] | None:
    conversation = state["conversation"]
    fitted = {"agent": state["agent"], "conversation": [], "user_message": state["user_message"]}
    # Serialize each entry once: a non-empty list dumps as "[" + entries joined by ", " + "]".
    lengths = [_size(entry) for entry in conversation]
    base = _size(fitted)
    total = base + sum(lengths) + 2 * (len(lengths) - 1) if lengths else base
    start = 0
    while total > budget and start < len(lengths):
        total -= lengths[start] + (2 if start < len(lengths) - 1 else 0)
        start += 1
    fitted["conversation"] = conversation[start:]
    size = _size(fitted)
    while size > budget and fitted["conversation"]:  # safety net in case the estimate is off
        fitted["conversation"].pop(0)
        size = _size(fitted)
    if size <= budget:
        return fitted
    text = state["user_message"]
    room = budget - _size({**fitted, "user_message": ""})
    while room > 1:
        head = room // 2
        tail = room - head - 1
        fitted["user_message"] = text[:head] + "…" + (text[len(text) - tail:] if tail else "")
        excess = _size(fitted) - budget
        if excess <= 0:
            return fitted
        room -= -(-excess // 6)  # a source char serializes to at most 6 chars (\uXXXX)
    return None
