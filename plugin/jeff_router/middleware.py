"""llm_execution middleware: ask Jeff in the background, pass the call to Strata unchanged, log both.

The agent thread never waits for Jeff: the record is appended from the Jeff future's done-callback. At most `slots`
Jeff requests are in flight per router; a call that finds no free slot is logged as local_busy instead of queueing."""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from .compare import _get, logged_action, pick, read_match, strata_action, strata_cost, tool_match
from .guard import failed_commands, first_runaway, repeated_action, stuck_outputs
from .jeff_client import JeffClient, JeffResult
from .options import (
    READ_INSTRUCTIONS, TOOL_INSTRUCTIONS, ReadOptions, ToolOptions, build_read_options, build_tool_options,
)
from .shadow_log import ShadowLog
from .state import build_state, fit_state
from .thinking import ask_level, render_state

logger = logging.getLogger(__name__)
_ENVELOPE_CHARS = 64  # {"model": "tools", "state": , "questions": } around state and questions


@dataclass
class _Pending:
    outcome: str | None = None
    cwd: str = ""
    tool: ToolOptions | None = None
    read: ReadOptions | None = None
    state: dict[str, Any] | None = None
    future: Future | None = None
    cost: dict[str, Any] | None = None
    strata_thinking: bool = True
    router: dict[str, Any] | None = None
    guard: dict[str, Any] | None = None


def _first_choice(response: Any) -> Any:
    choices = _get(response, "choices") if response is not None else None
    return choices[0] if choices else None


class ShadowRouter:
    def __init__(self, *, client: JeffClient, log: ShadowLog, cwd: Callable[[], str], max_request_chars: int,
                 slots: int = 2, strata_thinking: bool = True, thinking_router: bool = False,
                 router_adapter: str = "code-router", router_threshold: float = 0.6) -> None:
        self._client = client
        self._strata_thinking = strata_thinking
        self._thinking_router = thinking_router
        self._router_adapter = router_adapter
        self._router_threshold = router_threshold
        self._log = log
        self._cwd = cwd
        self._max_request_chars = max_request_chars
        self._slots = threading.BoundedSemaphore(slots)
        self._executor = ThreadPoolExecutor(max_workers=slots, thread_name_prefix="jeff_router")
        self._warned: set[str] = set()
        self._closed = False

    def __call__(self, request: Any = None, next_call: Any = None, **context: Any) -> Any:
        if self._closed or not self._gated(request, context):
            self._note_skipped_api_mode(context.get("api_mode"))
            return next_call()
        routed = self._strata_thinking and self._thinking_router
        router = self._route(request) if routed else None
        pending = self._start(request)
        pending.router = router
        messages = request.get("messages") or []
        forced = self._forced(messages) if routed else []
        if forced:
            pending.guard = {"forced": forced}
        thinking = self._strata_thinking and (bool(forced) or router is None or router["level"] != "off")
        started = time.monotonic()
        response = self._call(request, next_call, thinking)
        trigger = self._trigger(messages, response, thinking) if routed else None
        if trigger is not None:
            # Jeff-Code's thinking control: discard the reply and ask exactly once more, thinking on
            first_ms = round((time.monotonic() - started) * 1000)
            pending.guard = {**(pending.guard or {}), "trigger": trigger,
                             "discarded": self._safe_cost(request, response, first_ms)}
            thinking = True
            started = time.monotonic()
            response = self._call(request, next_call, thinking)
            pending.guard["reask_runaway"] = self._runaway(response)
        pending.strata_thinking = thinking
        self._finish(pending, request, response, round((time.monotonic() - started) * 1000), context)
        if not thinking and pending.cost is not None and pending.cost["reasoning_chars"] > 0:
            self._note_once(f"leak:{context.get('provider')}:{context.get('model')}",
                            "jeff_router: thinking was turned off but %s/%s still returned %d reasoning chars; "
                            "this route may ignore reasoning_effort=none",
                            context.get("provider"), context.get("model"), pending.cost["reasoning_chars"])
        return response

    def _note_skipped_api_mode(self, api_mode: Any) -> None:
        if api_mode is not None and api_mode != "chat_completions":
            self._note_once(f"api_mode:{api_mode}",
                            "jeff_router: api_mode %s is not supported yet; its calls pass through untouched", api_mode)

    def _note_once(self, key: str, message: str, *args: Any) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message, *args)

    @staticmethod
    def _call(request: dict[str, Any], next_call: Any, thinking: bool) -> Any:
        # Strata (llama.cpp) honours reasoning_effort "none"; it ignores extra_body.reasoning.enabled
        return next_call() if thinking else next_call({**request, "reasoning_effort": "none"})

    def _forced(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Jeff-Code forces thinking when the last outputs are stuck or the last commands failed. Never raises."""
        try:
            return [hit for hit in (stuck_outputs(messages), failed_commands(messages)) if hit]
        except Exception as error:
            self._warn("forced", error)
            return []

    def _runaway(self, response: Any) -> dict[str, Any] | None:
        """A runaway in the reply's thinking, then its text. Never raises."""
        try:
            message = _get(_first_choice(response), "message")
            for where, key in (("thinking", "reasoning_content"), ("thinking", "reasoning"), ("text", "content")):
                text = _get(message, key)
                hit = first_runaway(text) if isinstance(text, str) else None
                if hit:
                    return {"trigger": "runaway", "where": where, **hit}
        except Exception as error:
            self._warn("runaway", error)
        return None

    def _trigger(self, messages: list[Any], response: Any, thinking: bool) -> dict[str, Any] | None:
        """Runaway at any level; the loop guard only when thinking was off and the reply calls tools. Never raises."""
        runaway = self._runaway(response)
        if runaway is not None or thinking:
            return runaway
        try:
            choice = _first_choice(response)
            calls = _get(_get(choice, "message"), "tool_calls") or []
            if _get(choice, "finish_reason") != "tool_calls" or not calls:
                return None
            repeat = repeated_action(messages, list(calls))
            return {"trigger": "loop", **repeat} if repeat else None
        except Exception as error:
            self._warn("loop", error)
            return None

    def _safe_cost(self, request: Any, response: Any, ms: int) -> dict[str, Any] | None:
        try:
            return strata_cost(request, response, ms)
        except Exception as error:
            self._warn("cost", error)
            return None

    def close(self) -> None:
        """Plugin unload: stop admitting Jeff calls and drop queued work. Running HTTP calls finish and free their
        slots; their records are still written."""
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _route(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Jeff-Code's thinking router, asked synchronously before the call. Never raises: None means "not asked"."""
        try:
            level, result = ask_level(self._client, self._router_adapter, render_state(request.get("messages") or []),
                                      self._router_threshold)
            probabilities = (result.answers or {}).get("q", {}).get("probabilities") if result.outcome == "ok" else None
            return {"level": level, "outcome": result.outcome, "ms": result.ms, "probabilities": probabilities}
        except Exception as error:
            self._warn("route", error)
            return None

    @staticmethod
    def _gated(request: Any, context: dict[str, Any]) -> bool:
        count = context.get("api_call_count")
        tools = request.get("tools") if isinstance(request, dict) else None
        return (bool(context.get("turn_id")) and isinstance(count, int) and not isinstance(count, bool)
                and context.get("purpose") is None and context.get("api_mode") == "chat_completions"
                and isinstance(tools, list) and bool(tools))

    def _start(self, request: dict[str, Any]) -> _Pending:
        """Runs on the agent thread (session_cwd() is task-local). Never raises."""
        try:
            cwd = self._cwd() or ""
            messages = request.get("messages") or []
            tool = build_tool_options(request["tools"])
            read = build_read_options(messages, set(tool.names.values()), cwd)
            questions: dict[str, Any] = {"tool": {"type": "choice", "instructions": TOOL_INSTRUCTIONS,
                                                  "criteria": tool.criteria}}
            if read is not None:
                questions["read_target"] = {"type": "choice", "instructions": READ_INSTRUCTIONS,
                                            "criteria": read.criteria}
            budget = self._max_request_chars - len(json.dumps(questions, ensure_ascii=False)) - _ENVELOPE_CHARS
            state = fit_state(build_state(messages), budget) if budget > 0 else None
            pending = _Pending(cwd=cwd, tool=tool, read=read, state=state)
            if state is None:
                pending.outcome = "too_long"
                return pending
            if not self._slots.acquire(blocking=False):
                pending.outcome = "local_busy"
                return pending
            try:
                pending.future = self._executor.submit(self._ask, {"model": "tools", "state": state,
                                                                   "questions": questions})
            except BaseException:
                self._slots.release()
                raise
            return pending
        except Exception as error:
            self._warn("build", error)
            return _Pending(outcome="build_error")

    def _ask(self, body: dict[str, Any]) -> JeffResult:
        """Pool thread. The slot covers only the Jeff HTTP call: holding it through Strata's round trip would turn
        concurrent gateway turns into local_busy while Jeff sits idle. Logging afterwards never queues work, because
        the only lock it can meet is another writer's single append (the report reads without locking)."""
        try:
            return self._client.ask(body)
        finally:
            self._slots.release()

    def _finish(self, pending: _Pending, request: Any, response: Any, strata_ms: int,
                context: dict[str, Any]) -> None:
        """Runs on the agent thread after Strata answered. Never waits for Jeff, never raises."""
        try:
            pending.cost = strata_cost(request, response, strata_ms)
        except Exception as error:
            self._warn("cost", error)
        try:
            action = strata_action(response)
        except Exception as error:
            self._warn("finish", error)
            action = None
        if pending.future is not None:
            pending.future.add_done_callback(lambda future: self._complete(pending, future, action, context))
            return
        try:
            self._log.append(self._record(pending, pending.outcome or "bad_response", None, action, context))
        except Exception as error:
            self._warn("finish", error)

    def _complete(self, pending: _Pending, future: Future, action: dict[str, Any] | None,
                  context: dict[str, Any]) -> None:
        """Done-callback: runs in the pool thread when Jeff finishes, or inline if it already has. Appending is one
        short write under the writers' flock (readers take no lock), so running inline does not stall the turn."""
        if future.cancelled():  # dropped by close(); nothing to compare
            return
        try:
            try:
                result: JeffResult | None = future.result()
                outcome = result.outcome
            except Exception:
                result, outcome = None, "bad_response"
            self._log.append(self._record(pending, outcome, result, action, context))
        except Exception as error:
            self._warn("complete", error)

    @staticmethod
    def _record(pending: _Pending, outcome: str, result: JeffResult | None, action: dict[str, Any] | None,
                context: dict[str, Any]) -> dict[str, Any]:
        ok = outcome == "ok" and result is not None and result.answers is not None
        tool_block = read_block = None
        tool_ok = read_ok = None
        key, p = pick(result.answers.get("tool")) if ok else (None, None)
        if ok and key is None:  # "ok" without a usable tool answer: nothing to compare, but still one record
            outcome, ok = "bad_response", False
        if ok and pending.tool is not None:
            tool_block = {"options": pending.tool.criteria, "names": pending.tool.names,
                          "probabilities": result.answers["tool"]["probabilities"], "pick": key, "p": p,
                          "truncated_tools": pending.tool.truncated}
            tool_ok = tool_match(key, pending.tool.names, action)
            if pending.read is not None and "read_target" in result.answers:
                rkey, rp = pick(result.answers["read_target"])
                read_block = {"candidates": pending.read.paths,
                              "probabilities": result.answers["read_target"]["probabilities"], "pick": rkey, "p": rp}
                read_ok = read_match(rkey, pending.read.paths, action, pending.cwd)
        return {
            "v": 2,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "session_id": context.get("session_id"), "turn_id": context.get("turn_id"),
            "api_request_id": context.get("api_request_id"), "api_call_count": context.get("api_call_count"),
            "model": context.get("model"), "outcome": outcome,
            "jeff_ms": result.ms if result else None, "input_tokens": result.input_tokens if ok else None,
            "tool": tool_block, "read_target": read_block, "strata": logged_action(action),
            "strata_cost": pending.cost, "strata_thinking": pending.strata_thinking,
            "router": pending.router, "guard": pending.guard,
            "tool_match": tool_ok, "read_match": read_ok, "state": pending.state,
        }

    def _warn(self, where: str, error: Exception) -> None:
        key = f"{where}:{type(error).__name__}"
        if key in self._warned:
            logger.debug("jeff_router: %s failed again: %s", where, error)
            return
        self._warned.add(key)
        logger.warning("jeff_router: %s failed (%s: %s); the call continues unchanged and later errors of this "
                       "kind are logged at DEBUG", where, type(error).__name__, error)
