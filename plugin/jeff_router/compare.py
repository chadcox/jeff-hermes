"""Strata's actual next action, and whether Jeff's picks agree with it."""
from __future__ import annotations

import json
from typing import Any

from .options import normalize_path

MAX_ARGUMENT_CHARS = 2000  # per string value in the logged copy of Strata's call; write_file can carry a whole file


def _cap(value: Any) -> Any:
    """A copy of value with every string longer than MAX_ARGUMENT_CHARS cut and marked with '…'."""
    if isinstance(value, str):
        return value if len(value) <= MAX_ARGUMENT_CHARS else value[:MAX_ARGUMENT_CHARS] + "…"
    if isinstance(value, dict):
        return {key: _cap(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cap(item) for item in value]
    return value


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {"_raw": raw[:500]}
        return parsed if isinstance(parsed, dict) else {"_raw": raw[:500]}
    return {}


def strata_action(response: Any) -> dict[str, Any] | None:
    """Strata's next action with its first call's arguments intact; compare on this, log logged_action() of it."""
    choices = _get(response, "choices") if response is not None else None
    if not choices:
        return None
    message = _get(choices[0], "message")
    calls = _get(message, "tool_calls") or []
    if not calls:
        return {"kind": "text"}
    names = [_get(_get(call, "function"), "name") for call in calls]
    first_function = _get(calls[0], "function")
    return {"kind": "tool", "tools": names,
            "first": {"name": names[0], "arguments": _arguments(_get(first_function, "arguments"))}}


def strata_cost(request: Any, response: Any, ms: int) -> dict[str, Any]:
    """What Strata's answer cost: wall time, the reasoning setting sent, token usage, and the size of the reasoning
    and content text. Strata's usage has no reasoning-token count, so character counts stand in for the split."""
    choices = _get(response, "choices") if response is not None else None
    choice = choices[0] if choices else None
    message = _get(choice, "message")
    usage = _get(response, "usage") if response is not None else None
    extra = request.get("extra_body") if isinstance(request, dict) else None
    reasoning = _get(message, "reasoning_content") or _get(message, "reasoning")
    content = _get(message, "content")
    return {"ms": ms, "effort": extra.get("reasoning") if isinstance(extra, dict) else None,
            "finish_reason": _get(choice, "finish_reason"),
            "prompt_tokens": _get(usage, "prompt_tokens"), "completion_tokens": _get(usage, "completion_tokens"),
            "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
            "content_chars": len(content) if isinstance(content, str) else 0}


def logged_action(action: dict[str, Any] | None) -> dict[str, Any] | None:
    """The copy of a strata_action() result written to the log: first-call argument strings capped."""
    if action is None or action["kind"] != "tool":
        return action
    return {**action, "first": {**action["first"], "arguments": _cap(action["first"]["arguments"])}}


def pick(answer: Any) -> tuple[str | None, float | None]:
    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    if not isinstance(probabilities, dict) or not probabilities:
        return None, None
    key = max(probabilities, key=probabilities.get)
    return key, float(probabilities[key])


def tool_match(pick_key: str | None, names: dict[str, str], action: dict[str, Any] | None) -> bool | None:
    if pick_key is None or action is None:
        return None
    if pick_key in ("answer_directly", "ask_user"):
        return action["kind"] == "text"
    if pick_key not in names:
        return None
    if action["kind"] != "tool":
        return False
    actual = action["first"]["name"]
    if not isinstance(actual, str) or not actual:
        return None
    return names[pick_key] == actual


def read_match(pick_key: str | None, paths: dict[str, str], action: dict[str, Any] | None, cwd: str) -> bool | None:
    if pick_key is None or action is None or action["kind"] != "tool" or action["first"]["name"] != "read_file":
        return None
    target = normalize_path(action["first"]["arguments"].get("path"), cwd)
    if target is None:
        return None
    if pick_key == "none_of_these":
        return target not in paths.values()
    return paths.get(pick_key) == target
