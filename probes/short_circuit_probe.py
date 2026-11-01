"""Throwaway probe: can llm_execution middleware answer a turn's model call itself with a
synthetic tool call, have Hermes run the tool, and continue the loop? No network: both calls
are short-circuited."""
import json
from types import SimpleNamespace as NS

from hermes_cli.plugins import _delivery_manager
from run_agent import AIAgent

seen = []


def fake_response(message, finish):
    return NS(id="jeff-synth", model="jeff", object="chat.completion", usage=None,
              choices=[NS(index=0, message=message, finish_reason=finish)])


def jeff_mw(request, next_call, api_call_count=None, **ctx):
    msgs = request["messages"]
    seen.append({"call": api_call_count, "n_msgs": len(msgs), "last_role": msgs[-1]["role"],
                 "tools": len(request.get("tools") or [])})
    if msgs[-1]["role"] == "user":  # first call of the turn: Jeff "proposes" a read-only tool call
        tc = NS(id="call_jeff_1", type="function",
                function=NS(name="read_file", arguments=json.dumps({"path": "/etc/hostname"})))
        return fake_response(NS(role="assistant", content=None, tool_calls=[tc]), "tool_calls")
    # second call: the tool result must be the last message
    seen[-1]["tool_result"] = str(msgs[-1].get("content"))[:200]
    return fake_response(NS(role="assistant", content="SYNTH-FINAL", tool_calls=None), "stop")


_delivery_manager()._middleware.setdefault("llm_execution", []).append(jeff_mw)

agent = AIAgent(base_url="http://127.0.0.1:9/v1", model="probe-model", api_key="dummy",
                enabled_toolsets=["file"], max_iterations=5, quiet_mode=True)
result = agent.run_conversation("What is this machine's hostname?")
print("SEEN", json.dumps(seen, indent=1))
print("API_CALLS", result["api_calls"], "COMPLETED", result["completed"])
print("FINAL", result["final_response"])
for m in result["messages"]:
    print("MSG", m.get("role"), (m.get("tool_calls") and [t["function"]["name"] for t in m["tool_calls"]]),
          str(m.get("content"))[:120])


assert result["completed"] and result["api_calls"] == 2, result
assert [s["last_role"] for s in seen] == ["user", "tool"], seen
assert seen[1]["tool_result"].startswith('{"content": "1|'), seen
assert [m["role"] for m in result["messages"]] == ["user", "assistant", "tool", "assistant"]
assert result["final_response"] == "SYNTH-FINAL"
print("PROBE OK")