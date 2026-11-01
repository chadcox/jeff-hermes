# Step 1: can a plugin answer a model call before it reaches Strata?

Hermes checkout: `~/.hermes/hermes-agent` @ `bf0203a794`, VERSION 1.2.4. Provider: `strata`, `api_mode: chat_completions`.

## Answer: yes. No patch to Hermes is needed.

Use the **`llm_execution` middleware**, not a hook.

- `ctx.register_middleware("llm_execution", fn)` (`hermes_cli/middleware.py`, `VALID_MIDDLEWARE`).
- It wraps every main-loop provider call: `agent/turn_api_call.py::perform_api_call` →
  `run_llm_execution_middleware(api_kwargs, _perform_api_call, ...)`.
- The callback gets `request` (the full provider kwargs: `messages`, `tools`, ...) and `next_call`, plus context
  `turn_id`, `api_request_id`, `api_call_count`, `session_id`, `platform`, `model`, `provider`, `api_mode`, `task_id`.
- If it does not call `next_call`, its return value is used as the provider response (documented in
  `website/docs/developer-guide/plugins/index.md` under "Middleware": "skip it to short-circuit").
- The loop then validates the response (`ChatCompletionsTransport.validate_response`: needs non-empty `.choices`),
  normalizes it (`normalize_response`: `choices[0].message.{content,tool_calls}`, `finish_reason`; `usage` may be
  None), runs the tool calls, appends the results, and makes the next model call. That next call goes through the
  middleware again, so the plugin can let it reach Strata or answer it too.

So the Jeff-Code pattern looks like this: on each call, the plugin summarizes `request["messages"]` for Jeff, asks
Jeff to pick among candidate read-only tool calls, and if p > threshold, returns a synthetic assistant message with
that tool call. Otherwise it returns `next_call(request)`.

### Why not the hooks

- `pre_llm_call`: runs once per turn and can only add text to the user message. It cannot answer the call.
- `pre_api_request` / `post_api_request`: observers. Their return values are ignored.
- `pre_tool_call`: can only block a tool or change its arguments.

## Verified

`probes/short_circuit_probe.py` adds an `llm_execution` middleware to a real `AIAgent` (toolset `file`, isolated
`HERMES_HOME`, unreachable base_url, so nothing goes over the network):

- call 1 (last message `user`) → the middleware returns a synthetic `read_file {"path": "/etc/hostname"}` tool call
- Hermes ran the tool itself. On call 2 the last message was `tool` with content `1|myhost`
- call 2 → the middleware returns final text. `completed=True`, `api_calls=2`, and the transcript is
  `user → assistant(tool_calls=[read_file]) → tool → assistant`.

Run it with the Hermes runtime. `venv/bin/python` is stale (3.11) and missing `ruamel`:

```sh
P=~/Desktop/projects/jeff-hermes/probes/short_circuit_probe.py
cd /tmp && HERMES_HOME=$(mktemp -d) ~/.hermes/tools/python-3.14.9+20261021-linux-x64/bin/python3 -I -c \
  "import sys,runpy; sys.path.insert(0,'$HOME/.hermes/hermes-agent'); import hermes_bootstrap; \
   runpy.run_path('$P', run_name='__main__')"
```

The probe asserts the call sequence and transcript shape and exits nonzero on regression; success prints `PROBE OK`.

Response shape that works (chat_completions):

```python
NS(id=..., model="jeff", object="chat.completion", usage=None,
   choices=[NS(index=0, finish_reason="tool_calls",
               message=NS(role="assistant", content=None,
                          tool_calls=[NS(id="call_jeff_1", type="function",
                                         function=NS(name="read_file", arguments=json.dumps(args)))]))])
```

## Constraints for the real plugin

1. **Pass warm-compaction requests through untouched.** `~/.hermes/plugins/warm_compaction/engine.py` also calls
   `run_llm_execution_middleware`, with context `purpose="warm_compaction"`, `api_request_id=None`, and no
   `turn_id`. Its terminal refuses a request that has been changed (`middleware_rewrite`). Only act when
   `turn_id`/`api_call_count` are present and `purpose` is absent.
2. **Order relative to warm_compaction.** Its capture middleware has to be *last* in the `llm_execution` chain
   (`capture.py::_middleware_order_refusal`). Plugins are discovered alphabetically, so a plugin directory named
   `jeff_*` comes before `warm_compaction`. That order is correct. When Jeff answers a call itself, the capture never
   sees that request. [INFERENCE] warm compaction then skips that request. This is harmless, because that request
   never reached Strata.
3. **Prefix cache.** The synthetic assistant/tool messages are appended at the end of `messages`. The tool list and
   system prompt stay the same, so Strata's prefix cache keeps working on the next real call.
4. **Accounting.** A synthetic call counts toward `api_calls` and `max_iterations`. `pre_api_request`/
   `post_api_request` still fire, with `usage=None`. [INFERENCE] Cost and context-length tracking ignore a None usage.
   This needs checking when the real plugin is built.
5. **Read-only allow-list.** Only `read_file`, `search_files`, and similar tools may be auto-proposed. Approval
   (`pre_tool_call` and guardrails) still runs on synthetic tool calls, because the tools go through the normal
   execution path.
6. **Enablement.** A directory plugin in `~/.hermes/plugins/<name>/` (`plugin.yaml` + `__init__.py:register(ctx)`)
   must be listed in `plugins.enabled` in `~/.hermes/config.yaml`.
7. **Jeff's own latency counts against every model call it sees.** If p ≤ threshold, the time Jeff took is added
   to that call. Step 2's CPU latency number decides whether this works.
8. **Fail open to Strata.** Jeff is optional, and Hermes must never fail or stall a turn because of it. Every Jeff
   call needs a short client timeout, on the order of 150 ms on GPU. Any connection error, timeout, non-200 reply
   (including CUDA OOM or a server restart), or malformed answer means `return next_call(request)`, plus one log line.
   Jeff shares the GPU with Hindsight's Ollama (see step 2, "Coexistence"). A larger Ollama model could keep Jeff from
   starting or push it out, so being unavailable is an expected state, not an exceptional one.
