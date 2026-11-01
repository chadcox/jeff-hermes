# Configuration

All settings are optional. Put them under `plugins.entries.jeff_router.settings` in `~/.hermes/config.yaml` and
restart Hermes. An invalid value falls back to its default.

| Setting | Default | What it does |
|---|---|---|
| `enabled` | `true` | Master switch. `false` loads the plugin but registers nothing. |
| `url` | `http://127.0.0.1:8765` | Base URL of jeff-serve. |
| `timeout_s` | `2.0` | Client timeout for one Jeff request, in seconds. |
| `max_request_chars` | `24000` | Character budget for a Jeff request, which keeps it under jeff-serve's 8192-token limit. |
| `log_max_mb` | `50` | When `shadow.jsonl` reaches this size it is rotated to `shadow.jsonl.1`. |
| `thinking_router` | `false` | **The main feature.** Asks Jeff before each main-turn call whether to think. |
| `router_adapter` | `code-router` | The jeff-serve adapter that answers the thinking question. |
| `router_threshold` | `0.6` | The call thinks when Jeff's P(xhigh) is at least this value (0 to 1). Jeff-Code's evaluated setting is 0.6. |
| `main_thinking` | `true` | `false` turns thinking off on every main-turn call and overrides the router. |

`strata_thinking`, the v0.1 name for `main_thinking`, is still read when `main_thinking` is not set.

## Choosing a threshold

A lower threshold makes the main model think on more calls, which costs more but is safer. A higher threshold makes
it think on fewer calls. The loop guard and runaway guard (see [how-it-works.md](how-it-works.md)) catch the worst
cases of thinking too little. To compare thresholds, see what fraction of calls thought (the `main_thinking` field)
and whether the tasks still succeeded.

## Supported providers

The plugin works with **any provider on Hermes' `chat_completions` API mode**: local llama.cpp or vLLM, Ollama,
OpenRouter, and other OpenAI-compatible endpoints. It turns thinking off by sending `reasoning_effort: "none"`.

Calls in other API modes (`anthropic_messages`, `codex_responses`, `bedrock_converse`) pass through untouched and
are not logged. Hermes logs one warning per mode.

If a route ignores `reasoning_effort` and still returns reasoning text while thinking is off, the plugin logs one
warning per provider/model pair. That means the router can't save anything on that route.

## Failure behaviour

The plugin never breaks a turn:

- If jeff-serve is down, slow, busy or returns something unusable, the call thinks as usual. The record shows why:
  `router.outcome` is `refused`, `timeout`, `busy`, `bad_response` or `error`.
- If the plugin itself hits an error, it logs a warning and passes the call through unchanged.
- To disable the plugin, set `enabled: false` or remove it from `plugins.enabled`, then restart Hermes.

## The log

Path: `~/.hermes/plugin-data/agent-plugin-jeff_router-<digest>/shadow.jsonl`

The log has one JSON line per main-turn call. Its fields include:

- `router`: the level Jeff chose (`off` / `xhigh`), the outcome, the probabilities, and how long Jeff took.
- `guard`: whether thinking was forced (by stuck outputs or failed commands), or the reply was discarded and the call
  re-asked (runaway or loop).
- `main_thinking`: whether the call that was kept actually thought.
- `main_cost`: wall time, token usage, and reasoning and content character counts.
- `jeff` / `main`: Jeff's predicted next tool beside the tool the main model actually called (shadow comparison).

The log contains file paths and snippets of your transcripts, so it is created owner-only (`0600`). String values in
logged tool arguments are cut at 2,000 characters.

Summarise it with:

```sh
python tools/shadow_report.py              # every shadow.jsonl* in the plugin's data dir
python tools/shadow_report.py FILE ...     # specific files
```
