# Jeff router v1 (shadow mode): design

Status: approved 2026-10-30. Builds on `docs/research/01-hook-api.md` (hook point and constraints) and `docs/research/02-install-and-latency.md`
(install, CPU/GPU latency, coexistence with Hindsight's Ollama).

## Goal

Measure, on real Hermes traffic, how often Jeff's pick agrees with Strata's next action, and at what probability
threshold. Strata's action is a reference, not ground truth: a mismatch is not necessarily a Jeff error. v1 never
changes what Hermes does. Its output is a log and a report that set the threshold for v2, in which Jeff answers some
calls instead of Strata.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| v1 scope | Shadow only: observe and log, never short-circuit | Threshold and agreement on Hermes traffic are unknown; Jeff's README fixes thresholds on calibration data first |
| Jeff device | GPU (RTX 5080), `.venv-gpu` with `causal-conv1d` | 18–48 ms per decision, versus 0.3–1.5 s on CPU; coexists with Hindsight's Ollama (step 2) |
| Service lifecycle | Always on: systemd user unit, enabled at login (linger is on) | Holds about 2.2 GB of VRAM permanently; user chose this over socket activation |
| Plugin source | `plugin/jeff_router/` in this repo, symlinked to `~/.hermes/plugins/jeff_router` | Kept under git; Hermes discovery follows directory symlinks (`plugins_discovery.py:162`, `child.is_dir()`) |
| Plugin name | `jeff_router` | Sorts before `warm_compaction`, so its `llm_execution` middleware registers first (step 1, constraint 2) |
| Log content | Includes the state text Jeff saw | Allows offline re-scoring with other prompts or thresholds; local file only |

## Components

### 1. `jeff-serve` service

Unit file `deploy/jeff-serve.service` in this repo, installed to `~/.config/systemd/user/jeff-serve.service`:

- `ExecStart`: `<repo>/.venv-gpu/bin/jeff-serve`, with `WorkingDirectory=<repo>`.
- Environment:
  - `JEFF_DEVICE=cuda`
  - `JEFF_CHECKPOINT=models/jeff-base-v1.3`
  - `JEFF_ADAPTERS=models/adapters`
  - `PORT=8765`
  - `JEFF_HOST=127.0.0.1`
  - `JEFF_QUEUE_MS=250`
- `JEFF_QUEUE_MS` is how long a request waits for the model lock. It is not a batching window. With 250 ms, overlapping Hermes processes (gateway, desktop, CLI) queue briefly instead of getting 529 at once.
- `Restart=on-failure`, `RestartSec=5`, `WantedBy=default.target`.
- Ready when `GET /health` returns `{"status": "ready", ...}`.
- Warm-up: two `ExecStartPost` lines; the unit is reported started only after both finish, so the first real request
  does not pay about 1 s of CUDA warm-up, which with `JEFF_QUEUE_MS=250` would turn an overlapping request into 529.
  - Health wait: up to 120 polls of `/health` (each up to 2 s, then a 1 s sleep), bounded overall by
    `TimeoutStartSec=180`. If `/health` never reports ready, the start fails.
  - Example request: sends `models/adapters/tools/example.json` to `/v1/systemone`. It is best-effort (`-` prefix):
    its failure does not fail the unit.

### 2. Plugin `jeff_router`

Files under `plugin/jeff_router/`:

| File | Responsibility |
|---|---|
| `plugin.yaml` | Manifest: `manifest_version: 2`, `name: jeff_router`, `kind: standalone`, `config_schema` for the settings below. No `provides_middleware`: v2 manifests reject that key, and `register()` adds the `llm_execution` middleware without declaring it |
| `__init__.py` | `register(ctx)`: read settings once, build the shadow runner, register the middleware. If `ctx` lacks a callable `get_config` or `register_middleware`, or `ctx.state.data_dir` (e.g. `hermes plugins validate` copies only callable `PluginContext` members), it logs one warning and registers nothing |
| `middleware.py` | The `llm_execution` callback: gating, background Jeff call, pass-through, comparison, logging; never raises |
| `state.py` | Turn `request["messages"]` into the `tools` adapter's state; trim to the token budget |
| `options.py` | Build the `tool` options from `request["tools"]` and the `read_target` candidates |
| `jeff_client.py` | HTTP client for `POST /v1/systemone` with timeout; maps every failure to a typed outcome |
| `shadow_log.py` | Append JSONL records with size-based rotation |
| `compare.py` | Turn Strata's response into an action and compare it with Jeff's picks |

Settings, read once in `register()` via `ctx.get_config` (`plugins.entries.jeff_router.settings.<key>`):

| Key | Default | Meaning |
|---|---|---|
| `url` | `http://127.0.0.1:8765` | jeff-serve base URL |
| `timeout_s` | `2.0` | Client timeout for the Jeff request. It is off the critical path in v1 |
| `max_request_chars` | `24000` | Character budget for the whole serialized request (state plus questions), so it stays under 8,192 tokens (see "Token budget") |
| `log_max_mb` | `50` | Rotate `shadow.jsonl` to `shadow.jsonl.1` above this size, in megabytes (a positive int) |
| `enabled` | `true` | Off switch without removing the plugin |

#### Gating

Act only when all of these hold. Otherwise return `next_call(request)` and log nothing:

- `turn_id` is present and `api_call_count` is an int.
- `purpose` is absent. Warm compaction calls with `purpose="warm_compaction"` and no `turn_id`.
- `api_mode == "chat_completions"`.
- `request["tools"]` is a non-empty list.

#### Call flow

1. Build the state and options from `request`, on the agent thread, because `ctx.session_cwd()` reads task-local context. If this fails, record `outcome="build_error"` and skip Jeff.
2. Admission: each router holds a `BoundedSemaphore(2)` and its own `ThreadPoolExecutor(max_workers=2)`. If no slot is free (non-blocking acquire), record `outcome="local_busy"` and skip Jeff. Otherwise submit the Jeff request; the slot is released when the request finishes, whatever its result. Jeff work therefore never queues up and never runs stale.
3. `response = next_call()`, with no payload, so the request passes on unchanged. If `next_call` raises, the exception propagates unchanged and nothing is logged. The in-flight Jeff request finishes on its own and frees its slot. This keeps Hermes' provider-error handling intact.
4. Snapshot Strata's action from `response.choices[0].message` (see Comparison) on the agent thread.
5. If Jeff was not asked, append the record now. Otherwise attach a done-callback to the Jeff future. The callback runs when Jeff finishes (in the pool thread, or at once if already done), builds the record and appends it.
6. Return `response`, the same object, without waiting for Jeff. The plugin adds no latency to any Strata call; its only cost on the agent thread is building the request: string work, at most `MAX_PATH_PROBES = 64` path probes (`stat`), and resolving the paths of the newest 200 `read_file` calls (see "`read_target` candidates").

Steps 1, 2, 4 and 5 are wrapped in `try/except Exception` and log one WARNING per distinct error type (rate-limited, as in `conduit_push`). The middleware can never raise its own exception into Hermes.

#### Questions sent to Jeff (one request, `model: "tools"`)

**`tool`** (choice), with the instruction text from the adapter card, word for word:

> Which tool should the agent call next to handle the user's latest message? Use the conversation for context. If no
> tool is needed, choose answer directly. If a tool is needed but information it requires is missing, choose ask the
> user.

Criteria, in this order:
- `answer_directly`: "No tool is needed: answer the user directly from the conversation"
- `ask_user`: "A tool is needed but required information is missing: ask the user first"
- `t1…tN`: every tool in `request["tools"]`, in request order, each as `"<name>(<required params, comma-separated>): <first sentence of description, ≤ 120 chars>"`.
- The adapter card gives 2–150 tools in training, on top of the two fixed options, and the adapter allows 252 options. If there are more than 150 tools (`MAX_TOOLS = 150`), the 150 kept are those in `READ_ONLY_TOOLS` first, then the rest in request order, and `truncated_tools: true` is recorded.
- `READ_ONLY_TOOLS` is a copy of Hermes' `NO_EFFECT_TOOL_NAMES` (`agent/tool_result_classification.py:16-20`): `read_file`, `search_files`, `session_search`, `skill_view`, `skills_list`, `web_extract`, `web_search`, `vision_analyze`, `browser_snapshot`, `browser_get_images`, `browser_console`, `read_terminal`. It is copied rather than imported, so the plugin and its tests do not import Hermes internals. The report uses the same set.

**`read_target`** (choice), asked only when at least one candidate exists. This question is experimental: the adapter was trained only on `tool`, and the report shows whether `read_target` is useful.

> Which file should the agent read next to handle the user's latest message? If none of these files is the right one,
> choose none of these.

Criteria:
- `r1…rK`: `"read_file(path=\"<abs path>\"): read <basename>"`, at most 10 candidates.
- `none_of_these`: "None of these: the agent needs a different file or a different tool".

#### `read_target` candidates

Text is scanned from the latest user message, then from the last 12 non-system messages (user, assistant and tool;
`SCAN_MESSAGES = 12`), newest first. A path qualifies if it matches one of these forms, and is not part of a URL:
- absolute: `/…`, one segment or more (`/x.md` qualifies)
- home-relative: `~/…`, one segment or more (`~/.bashrc` qualifies)
- relative, `./…` or `../…` (one segment or more), or bare `name/…` (at least one slash), resolved against
  `ctx.session_cwd()` when that is available

Trailing `.,:;` is stripped. Each distinct raw path is probed once, and the scan stops after
`MAX_PATH_PROBES = 64` probes or when 10 candidates are found (`MAX_READ_CANDIDATES = 10`).

A candidate is kept only if:
- it resolves to an existing regular file of 2 MB or less;
- none of the newest 200 `read_file` calls in `messages` (`MAX_ALREADY_READ = 200`, newest first) used the same `path` with no `offset`/`limit`, because Hermes refuses identical re-reads (`tool_result_classification.py:27-31`). Older calls are not checked, which bounds the path resolution done on the agent thread;
- `read_file` is present in `request["tools"]`.

Candidates are de-duplicated by resolved path, and at most 10 are kept.

#### State (`tools` adapter layout)

- **`agent`**: `"A personal assistant with shell, file, web, memory and other tools on the user's Linux workstation."`
- **`conversation`**: the last 6 entries before the latest user message, plus every step after it, rendered as `{role, text}`.
  - User messages are rendered as text. List content is reduced to its text parts.
  - Assistant messages are rendered as text. If there are tool calls, each becomes a line `-> name(args-json)`, cut to 300 characters.
  - Tool results become `{role: "tool", text: last 40 lines, each ≤ 200 chars}`.
  - System and developer messages are excluded.
- **`user_message`**: the latest `role: "user"` message's text.

#### Token budget

`jeff-serve` refuses requests over 8,192 tokens (422) and does not truncate. The questions are serialized first. The state then gets the remaining budget, `max_request_chars − len(questions JSON)`, and is trimmed to fit: drop the oldest `conversation` entries first, then cut `user_message` to its first and last `budget/2` characters. If the questions alone exceed the budget, nothing is sent and `outcome="too_long"` is recorded. The default of 24,000 characters assumes about 3 characters per token or more, which leaves headroom. A 422 for length is still logged as `outcome="too_long"`, and the report shows whether the budget needs tuning.

### Comparison

Strata's action, taken from `response.choices[0].message`:
- **Has `tool_calls`:** `kind="tool"`, `tools=[names in order]`, `first={name, arguments(parsed JSON)}`.
  - The match flags below compare on these arguments as parsed. Only the copy written to the log record's `strata` field is capped: every string value inside its `first.arguments` (at any depth) is cut to 2,000 characters, with `…` appended when cut (`MAX_ARGUMENT_CHARS = 2000`), so a `write_file` call does not copy a whole file into the log. The dict shape is kept. Arguments that are not a JSON object are recorded as `{"_raw": <first 500 chars>}`.
- **Otherwise:** `kind="text"`.

Match flags (each `true`, `false`, or `null` when it cannot be decided):
- `tool_match`:
  - Jeff picked `answer_directly` or `ask_user`, and `kind == "text"`; or
  - Jeff picked `tK`, and the name of `tK` equals `first.name`.
  - For multiple tool calls, only the first counts; all names are recorded.
  - `null` when Jeff has no pick, Strata's action is missing, the pick is not a known option, or Strata's first call has no name.
- `read_match` (only when `read_target` was asked and `first.name == "read_file"`): Jeff's pick is `rK` and its path equals `first.arguments.path` after resolving.
  - `none_of_these` counts as a match when Strata read a path outside the candidates.
  - `null` when Strata's first call is not `read_file` (text, or another tool), or its path cannot be resolved (missing, or relative with no cwd).

### Log record (`$HERMES_HOME/plugin-data/agent-plugin-jeff_router-<digest>/shadow.jsonl`, Hermes' namespaced plugin data dir, via `ctx.state.data_dir`)

One JSON object per gated call, written with a single `write()` in append mode, so concurrent processes don't interleave:

```json
{"v": 1, "ts": "2026-10-30T22:00:00Z", "session_id": "...", "turn_id": "...", "api_request_id": "...",
 "api_call_count": 2, "model": "swift-1.5-iq3_s", "outcome": "ok", "jeff_ms": 27.4, "input_tokens": 612,
 "tool": {"options": {"answer_directly": "...", "t1": "read_file(path): ..."}, "names": {"t1": "read_file"},
          "probabilities": {"answer_directly": 0.1, "t1": 0.8}, "pick": "t1", "p": 0.8, "truncated_tools": false},
 "read_target": {"candidates": {"r1": "/home/user/x.md"}, "probabilities": {"r1": 0.9, "none_of_these": 0.1},
                 "pick": "r1", "p": 0.9},
 "strata": {"kind": "tool", "tools": ["read_file"], "first": {"name": "read_file", "arguments": {"path": "/home/user/x.md"}}},
 "tool_match": true, "read_match": true, "state": {"agent": "...", "conversation": [], "user_message": "..."}}
```

- `outcome` is one of: `ok`, `local_busy` (both Jeff slots in use), `timeout`, `refused` (connection error), `busy` (529), `too_long` (questions over budget, or 422 for length), `bad_request` (other 4xx), `server_error` (5xx), `bad_response`, `build_error`.
- Records are written when Jeff finishes, so lines can appear slightly out of call order. `api_call_count` and `ts` identify each call.
- When the outcome is not `ok`, Jeff fields are null and the Strata fields are still filled, except `jeff_ms`: when Jeff was asked, it records how long the failed attempt took (null for `local_busy`, `build_error`, and `too_long` before sending).
- Permissions: the log, its `.1` and `shadow.jsonl.lock` are created owner-only (0600; an existing file with a wider mode is tightened with `fchmod`), because they hold paths and transcript snippets. A missing data directory is created 0700.
- Rotation: before appending, if the file is larger than `log_max_mb`, rename it to `shadow.jsonl.1`, replacing any older one. The gateway, desktop app and CLI are separate processes writing the same file, so rotate and append run under an exclusive `fcntl.flock` on `shadow.jsonl.lock`.

### 3. Report script `tools/shadow_report.py`

Usage: `.venv/bin/python tools/shadow_report.py [path …]`. The default path is `~/.hermes/plugin-data/agent-plugin-jeff_router-<digest>/shadow.jsonl*`, Hermes' namespaced plugin data dir: with no paths the script globs `$HERMES_HOME/plugin-data/*jeff_router*/` (HERMES_HOME defaults to `~/.hermes`); one match is read, none prints "No records" and exits 1, several are listed and it exits 1 asking for an explicit path. It prints Markdown:

- **Counts:** records, the share of each `outcome`, latency p50/p95 of `jeff_ms` for `ok` records, and the share of Strata `tool` versus `text` actions.
- **Tool choice: agreement with Strata**, for each threshold in 0.3, 0.4, …, 0.9:
  - coverage: the share (and count) of `ok` records with `tool.p ≥ t`;
  - agreement: `tool_match` among the covered records that could be compared (`tool_match` not null; count shown);
  - the same coverage and agreement restricted to picks of read-only tools.
- **Full `read_file` call and calls that could be skipped:** a t × t2 grid over records where Jeff picked `read_file` with `tool.p ≥ t` (rows) and `read_target.pick` is `rK` with `read_target.p ≥ t2` (columns), `t2` in the same thresholds. Each cell shows counts, not shares: `answered / match / mismatch`.
  - answered: calls Jeff would have answered locally; coverage as a share is in the tool table above.
  - match: `read_match` is true, i.e. Strata's first call was `read_file` with that path.
  - mismatch: `read_match` is false, or `read_match` is null and `tool_match` is false (Strata answered in text or called another tool, so Jeff's `read_file` call is one Strata did not make).
  - The rest of answered are undecided: Strata called `read_file` with a path that could not be resolved, or its action could not be read.
- Mismatches are not necessarily Jeff errors; the report says to review a sample of high-confidence mismatches by hand before trusting any threshold.

### 4. Tests

pytest under `tests/`, following the conventions of the hindsight plugin (`~/.hermes/plugins/hindsight/tests/conftest.py`):
- Load the plugin directory as a package with `importlib.util.spec_from_file_location`.
- Use a `SimpleNamespace` ctx.
- Serve fake Jeff responses from a real local HTTP server on an ephemeral port.

Tests cover:
- the gating rules;
- that the response object is returned unchanged;
- that a `next_call` exception propagates unchanged, with no log;
- each failure outcome;
- the `tool` options (order, signature format, the 150 cap);
- `read_target` candidates (existence, re-read exclusion, cwd resolution, cap of 10);
- state rendering and trimming;
- comparison flags;
- log rotation.

End-to-end check: extend `probes/short_circuit_probe.py`'s approach into `probes/shadow_probe.py`. It runs a real `AIAgent` with an isolated `HERMES_HOME`, the plugin registered in-process, and a fake provider. It asserts that one JSONL record per call is written and that the transcript is unchanged. `git status` in `~/.hermes/hermes-agent` must stay clean.

## Go-live

1. Install and enable `jeff-serve.service`, and check `/health`.
2. Symlink the plugin and add `jeff_router` to `plugins.enabled` in `~/.hermes/config.yaml`.
3. Restart `hermes-gateway.service`, only after asking the user. Restart the desktop app or CLI sessions as the user prefers.
4. Run one real turn and confirm that a record with `outcome="ok"` appears.

## Out of scope for v1 (v2 candidates)

- Answering calls instead of Strata. When this is built:
  - Strata's action is a reference, not ground truth. Before any threshold is switched on, review a sample of high-confidence mismatches by hand. Jeff's `answer_directly` and `ask_user` both count as agreeing with any Strata text reply, so text decisions are only coarsely scored.
  - Responses must pass `validate_response`.
  - The client timeout drops to about 150 ms.
  - Any failure falls back to Strata.
  - The warm_compaction capture conflict must be handled first: a call Jeff answers replaces warm_compaction's latest capture with a body-less one (`capture.py:186-190,255-286`).
- `search_files` argument candidates.
- `code-router` reasoning-effort routing.

## Risks

- **VRAM:** switching Hindsight back to `gpt-oss:20b`, re-enabling the local `qwen.service` (about 13.4 GB, currently disabled), ComfyUI, or games can exhaust VRAM. Jeff then fails to start or returns 5xx. The plugin logs the outcome and Hermes is unaffected.
- **Mid-turn calls:** the `tools` adapter was trained on single conversation-level decisions of 6 turns or fewer. Mid-turn calls, after tool results, are outside that range, and the report shows how much this matters.
- **Log contents:** the log holds paths and transcript snippets. It stays local under `$HERMES_HOME/plugin-data/`.
