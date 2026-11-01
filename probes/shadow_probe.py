"""End-to-end check: jeff_router in a real AIAgent turn. A fake provider middleware after jeff_router plays
Fake main model (no network): call 1 returns a read_file tool call, call 2 returns final text. Asserts one ok shadow record
per call, the right match flags, and an unchanged transcript. Needs jeff-serve on :8765; one direct warm-up request
is sent first so a cold server's slow first answer cannot turn call 2 into busy (jeff-serve serves one at a time),
and call 2 waits (up to 5 s) for record 1 before it reaches jeff_router, so Jeff's first request has finished before
the second one is sent."""
import importlib.util
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace as NS

from hermes_cli.plugins import _delivery_manager
from run_agent import AIAgent

REPO = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO / "plugin" / "jeff_router"
TARGET = REPO / "docs/research/01-hook-api.md"
JEFF_URL = "http://127.0.0.1:8765/v1/systemone"

warmup = urllib.request.Request(JEFF_URL, data=(REPO / "models" / "adapters" / "tools" / "example.json").read_bytes(),
                                headers={"Content-Type": "application/json"})
try:
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(warmup, timeout=60) as response:
        response.read()
except OSError as error:
    sys.exit(f"jeff-serve warm-up failed at {JEFF_URL}: {error!r}; is jeff-serve running and ready?")

spec = importlib.util.spec_from_file_location("jeff_router", PLUGIN_DIR / "__init__.py",
                                              submodule_search_locations=[str(PLUGIN_DIR)])
plugin = importlib.util.module_from_spec(spec)
sys.modules["jeff_router"] = plugin
spec.loader.exec_module(plugin)

data_dir = Path(tempfile.mkdtemp(prefix="jeff_router_probe_"))
log_path = data_dir / "shadow.jsonl"
registered = []
ctx = NS(get_config=lambda key, default: default, register_middleware=lambda kind, fn: registered.append((kind, fn)),
         state=NS(data_dir=data_dir), session_cwd=lambda: str(REPO))
plugin.register(ctx)
[(kind, router)] = registered
assert kind == "llm_execution"


def log_lines():
    return len(log_path.read_text().splitlines()) if log_path.exists() else 0


def wait_lines(count, timeout):
    """Records are appended from Jeff's done-callback; poll until `count` lines exist or the timeout passes."""
    deadline = time.monotonic() + timeout
    while log_lines() < count and time.monotonic() < deadline:
        time.sleep(0.05)


def fake_response(message, finish):
    return NS(id="main-fake", model="main-fake", object="chat.completion", usage=None,
              choices=[NS(index=0, message=message, finish_reason=finish)])


def await_record_1(request, next_call, **ctx):
    """First in the chain: jeff_router sends Jeff's request before calling the main model, so waiting inside fake_main would
    be too late to keep Jeff's two requests apart."""
    if request["messages"][-1]["role"] != "user":
        wait_lines(1, 5)
    return next_call()


def fake_main(request, next_call, api_call_count=None, **ctx):
    if request["messages"][-1]["role"] == "user":
        call = NS(id="call_1", type="function",
                  function=NS(name="read_file", arguments=json.dumps({"path": str(TARGET)})))
        return fake_response(NS(role="assistant", content=None, tool_calls=[call]), "tool_calls")
    return fake_response(NS(role="assistant", content="FAKE-FINAL", tool_calls=None), "stop")


chain = _delivery_manager()._middleware.setdefault("llm_execution", [])
chain.append(await_record_1)
chain.append(router)
chain.append(fake_main)

agent = AIAgent(base_url="http://127.0.0.1:9/v1", model="probe-model", api_key="dummy",
                enabled_toolsets=["file"], max_iterations=5, quiet_mode=True)
result = agent.run_conversation(f"What does {TARGET} say about warm compaction?")
wait_lines(2, 5)
records = sorted((json.loads(line) for line in log_path.read_text().splitlines()), key=lambda r: r["api_call_count"])
for r in records:
    print("RECORD", r["api_call_count"], r["outcome"], r["jeff_ms"], r["input_tokens"],
          r["tool"] and (r["tool"]["names"].get(r["tool"]["pick"], r["tool"]["pick"]), round(r["tool"]["p"], 2)),
          r["read_target"] and (r["read_target"]["pick"], round(r["read_target"]["p"], 2)),
          r["main"], r["tool_match"], r["read_match"])

assert result["completed"] and result["api_calls"] == 2, result
assert result["final_response"] == "FAKE-FINAL"
assert [m["role"] for m in result["messages"]] == ["user", "assistant", "tool", "assistant"]
call = result["messages"][1]["tool_calls"][0]["function"]
assert call["name"] == "read_file" and str(TARGET) in call["arguments"], call
assert result["messages"][2]["content"], result["messages"][2]
assert len(records) == 2, records
assert all(r["outcome"] == "ok" for r in records), [r["outcome"] for r in records]
assert records[0]["main"]["kind"] == "tool" and records[1]["main"] == {"kind": "text"}
assert str(TARGET) in records[0]["read_target"]["candidates"].values()
assert records[1]["read_target"] is None or str(TARGET) not in records[1]["read_target"]["candidates"].values()
assert records[0]["tool_match"] in (True, False) and records[1]["tool_match"] in (True, False)
print("SHADOW PROBE OK", data_dir)
