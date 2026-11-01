"""Minimal jeff-serve client. Every failure becomes an outcome string; ask() never raises."""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JeffResult:
    outcome: str
    answers: dict[str, Any] | None = None
    input_tokens: int | None = None
    ms: float | None = None
    detail: str = ""


def _http_outcome(code: int, body: str) -> str:
    if code == 529:
        return "busy"
    if code == 422 and "token limit" in body:
        return "too_long"
    if 400 <= code < 500:
        return "bad_request"
    return "server_error"


def _valid_answers(payload: Any) -> bool:
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict) or not answers:
        return False
    for answer in answers.values():
        probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
        if not isinstance(probabilities, dict) or not probabilities:
            return False
        if not all(isinstance(p, (int, float)) and not isinstance(p, bool) for p in probabilities.values()):
            return False
    return True


class JeffClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.timeout_s = timeout_s
        # No proxies: http_proxy/HTTP_PROXY must never reroute the local jeff-serve call.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def ask(self, body: dict[str, Any]) -> JeffResult:
        try:
            request = urllib.request.Request(self.url, json.dumps(body).encode("utf-8"),
                                             {"Content-Type": "application/json"}, method="POST")
        except (TypeError, ValueError, RecursionError) as error:
            return JeffResult("bad_request", detail=f"{type(error).__name__}: {error}")
        started = time.perf_counter()

        def elapsed() -> float:
            return round((time.perf_counter() - started) * 1000, 1)

        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            try:
                text = error.read().decode("utf-8", "replace")
            except Exception:
                text = ""
            finally:
                error.close()  # HTTPError owns the response socket; leaving it open leaks it.
            return JeffResult(_http_outcome(error.code, text), ms=elapsed(), detail=f"HTTP {error.code}")
        except urllib.error.URLError as error:
            outcome = "timeout" if isinstance(error.reason, TimeoutError) else "refused"
            return JeffResult(outcome, ms=elapsed(), detail=type(error.reason).__name__)
        except TimeoutError:
            return JeffResult("timeout", ms=elapsed(), detail="read timeout")
        except OSError as error:
            return JeffResult("refused", ms=elapsed(), detail=type(error).__name__)
        except http.client.HTTPException as error:
            return JeffResult("server_error", ms=elapsed(), detail=type(error).__name__)
        ms = elapsed()
        try:
            payload = json.loads(raw)
        except ValueError:
            return JeffResult("bad_response", ms=ms, detail="non-JSON body")
        if not _valid_answers(payload):
            return JeffResult("bad_response", ms=ms, detail="missing answers/probabilities")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        tokens = usage.get("input_tokens")
        return JeffResult("ok", answers=payload["answers"],
                          input_tokens=tokens if isinstance(tokens, int) else None, ms=ms)
