"""jeff_router: Jeff thinking router for Hermes Agent (see docs/ in the jeff-hermes repo)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)
CONTEXT_METHODS = ("get_config", "register_middleware")


def _cwd_reader(ctx: Any) -> Callable[[], str]:
    session_cwd = getattr(ctx, "session_cwd", None)
    if not callable(session_cwd):
        return lambda: ""
    return lambda: session_cwd() or ""


def register(ctx: Any) -> None:
    """Hermes entry point: register the shadow router as llm_execution middleware."""
    from .jeff_client import JeffClient
    from .middleware import ShadowRouter
    from .settings import read_settings
    from .shadow_log import ShadowLog

    missing = [name for name in CONTEXT_METHODS if not callable(getattr(ctx, name, None))]
    try:
        data_dir = ctx.state.data_dir
    except AttributeError:  # e.g. `hermes plugins validate` copies only callable PluginContext members
        missing.append("state.data_dir")
    if missing:
        logger.warning("jeff_router: Hermes plugin context lacks %s; registering nothing", ", ".join(missing))
        return
    settings = read_settings(ctx.get_config)
    if not settings["enabled"]:
        logger.info("jeff_router: disabled in settings; registering nothing")
        return
    log = (ShadowLog(Path(data_dir) / "shadow.jsonl", max_bytes=int(settings["log_max_mb"] * 1024 * 1024))
           if settings["log"] else None)
    router = ShadowRouter(client=JeffClient(settings["url"], settings["timeout_s"]), log=log, cwd=_cwd_reader(ctx),
                          max_request_chars=int(settings["max_request_chars"]),
                          main_thinking=settings["main_thinking"], thinking_router=settings["thinking_router"],
                          guards=settings["guards"], tool_shadow=settings["tool_shadow"],
                          router_adapter=settings["router_adapter"], router_threshold=settings["router_threshold"])
    ctx.register_middleware("llm_execution", router)
    on_unload = getattr(ctx, "on_unload", None)
    if callable(on_unload):
        on_unload(router.close)
    logger.info("jeff_router: shadow mode on; Jeff at %s, log %s", settings["url"], log.path if log else "off")


__all__ = ["register"]
