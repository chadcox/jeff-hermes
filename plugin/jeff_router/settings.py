"""jeff_router settings: read once in register(); invalid values fall back to the defaults."""
from __future__ import annotations

import math
from typing import Any, Callable

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "url": "http://127.0.0.1:8765",
    "timeout_s": 2.0,
    "max_request_chars": 24000,
    "log_max_mb": 50,
    "main_thinking": True,
    "thinking_router": False,
    "router_adapter": "code-router",
    "router_threshold": 0.6,
}

_UNSET = object()


def _valid(default: Any, value: Any) -> bool:
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if isinstance(default, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 and math.isfinite(value)
    if isinstance(default, str):
        return isinstance(value, str) and bool(value.strip())
    return False


def read_settings(get_config: Callable[[str, Any], Any]) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for key, default in DEFAULTS.items():
        try:
            value = get_config(key, _UNSET)
            if value is _UNSET and key == "main_thinking":  # v0.1 name
                value = get_config("strata_thinking", _UNSET)
            if value is _UNSET:
                value = default
        except Exception:
            value = default
        settings[key] = value if _valid(default, value) else default
    if settings["router_threshold"] > 1:
        settings["router_threshold"] = DEFAULTS["router_threshold"]
    return settings
