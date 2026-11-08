<div align="center">

# 🧠⚡ jeff-hermes

### Think only when it matters

**A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that lets a tiny 0.8B model decide, call by
call, whether your big model needs to think.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Hermes plugin](https://img.shields.io/badge/Hermes_Agent-plugin-8A2BE2)
![Jeff 0.8B](https://img.shields.io/badge/router-Jeff_0.8B-orange)
[![CI](https://github.com/chadcox/jeff-hermes/actions/workflows/ci.yml/badge.svg)](https://github.com/chadcox/jeff-hermes/actions/workflows/ci.yml)

</div>

---

> [!WARNING]
> **Research project.** It works best on slow models that think at length (see [Does it work?](#-does-it-work)), and
> it has only been tested on Qwen3.8 models so far. Settings, log formats and behaviour may still change between
> releases.

Reasoning models think on **every** step of an agent loop, including steps like `ls`, `cat` and "run the tests
again". That thinking costs tokens and wall-clock time, and most of those steps don't need it.

**jeff-hermes** ports the thinking router from [Jeff-Code](https://github.com/firelex/jeff-code) to Hermes. Before
each model call, [Jeff](https://github.com/firelex/jeff), a 0.8B model running locally, reads where the task is and
answers one question: *does this step need thinking?* If not, the call goes out with `reasoning_effort: "none"`. The
big model still does all the work. It just doesn't spend thinking tokens on easy steps.

```
 task + recent steps ──▶  Jeff (0.8B)  ──▶  P(xhigh) ≥ threshold ?
                                            ├─ yes ─▶  big model, thinking ON   🧠
                                            └─ no  ─▶  big model, thinking OFF  ⚡
```

## ✨ Features

- **⚡ Faster, cheaper turns.** Easy steps skip thinking entirely.
- **🛡️ Safety net.** Ported from Jeff-Code's loop and runaway guards. Thinking is forced back on when tool outputs
  get stuck or commands keep failing. Runaway and looping replies are detected and logged. Hermes allows only
  one provider call per middleware, so they can't be re-asked.
- **🔌 Built for `chat_completions` providers.** Tested with local llama.cpp. vLLM, Ollama, OpenRouter and other
  OpenAI-compatible servers on Hermes' `chat_completions` API mode should work but are untested. Other API modes pass
  through and are logged.
- **🧯 Fails safe.** If Jeff is down, slow or busy, the call thinks as normal. The plugin never breaks a turn.
- **📊 Fully logged.** Every call records Jeff's decision, its probabilities, the guards, and the main model's cost
  (time, tokens, reasoning characters) in an owner-only JSONL log, with a report tool to summarise it.
- **🪶 No changes to Hermes.** It is a plain plugin: one symlink and a few config lines.

## 🎯 Is it right for you?

**Where it helps**

- **Slow models that think at length.** On Qwen3.8 27B, runs took **about a third less time** than always-think, with
  **67–77% less reasoning text** and **40–50% fewer completion tokens**.
- **Local models where your time is the cost.** With llama.cpp or vLLM and prompt caching, the extra steps it causes
  cost little, and the time it saves is real.
- **Coding agents.** Jeff's `code-router` adapter was trained on coding-agent traces from a Qwen 27B.
- **Keeping quality close to always-think.** It passed 37/40 and 34/40 runs on the two 27B quants (always-think: 40/40
  and 37/40), and beat never-think on every model tested.
- **Catching stuck runs.** The guards force thinking back on when outputs get stuck, commands keep failing or the
  model loops. You can also use them on their own with `main_thinking: false`.

**What it can't do**

- **Save tokens.** With thinking off, the model takes more steps, and each step resends the conversation. Total
  tokens came out level with or higher than always-think, so it won't cut a pay-per-token API bill.
- **Speed up fast models.** On a model that thinks only briefly (Qwen3.8 Flash Next), there is little thinking to
  skip, and the median run took the same time as always-think.
- **Match always-think exactly.** It gives up a few points of pass rate. For work where every pass counts, keep
  thinking on.
- **Work without a thinking switch.** The model has to honour `reasoning_effort: "none"` (hybrid models such as Qwen3).
- **Promise results beyond what was tested.** Only Qwen3.8 models and eight coding tasks have been tested so far, and
  the 0.6 threshold hasn't been tuned for other models.

## 📈 Does it work?

Upstream, on Jeff-Code (Qwen 27B coding agent, 1,242 paired tasks), the `code-router` adapter turned thinking off on
**about 3/4 of turns**. The pass rate stayed within noise (**62.4% vs 62.8%**), and Jeff-Code reports **0.68× time
per task**.

On Hermes, with Strata running Qwen3.8 Flash Next (IQ3_S quant) — a fast local model that thinks only briefly — we ran an A/B with hidden-test grading: five coding
tasks × five reps × three modes, 75 runs in total.

| Mode | Runs fully passing | Thinking turns | Model calls | Prompt tokens | Completion tokens | Reasoning chars | Median wall time per run |
|---|---|---|---|---|---|---|---|
| Never think | 19/25 | 0% | 469 | 21.0M | 123k | 0 | 35.6 s |
| **Router @ 0.6** | **20/25** | **19%** | **225** | **6.3M** | **90k** | **57k** | 47.2 s |
| Always think | 19/25 | 100% | 150 | 4.2M | 118k | 195k | 47.6 s |

- **Quality held.** The router's pass rate matches always-think (within noise at n=5).
- **Less thinking, more steps.** Compared with always-think, the router cut reasoning text by about 70% and completion
  tokens by about 24%. But without thinking the model took more steps (225 calls vs 150), and each step resends the
  conversation, so prompt tokens rose by about 50%. On a pay-per-token API that likely costs more, not less.
- **Wall time didn't improve.** The router's median time was the same as always-think's. Jeff itself added about 50 ms
  per call.
- **The guards matter.** Never-think had two runs that got stuck in loops (150 and 76 model calls; this also inflates
  its prompt-token total). Under the router, the stuck-output guard fired twice and forced thinking, and neither run
  looped. The guards are simple rules and don't depend on Jeff.
- The threshold of 0.6 looks reasonable, but five reps per task can't tell nearby thresholds apart.

**Takeaway:** on a fast model that thinks briefly there is little thinking to skip, so the router keeps quality but
doesn't save time. Never-think plus the guards is a good setup there. The router is built for slower models that think
at length on every turn, which is where the upstream 0.68× result came from. The 27B runs below test that.

### Qwen3.8 27B (2026-11-04)

We then ran the same A/B on two quants of the dense Qwen3.8 27B, a slower model that thinks at length: eight coding
tasks × five reps × three modes per quant (240 runs), with hidden-test grading.

| Model | Mode | Runs fully passing | Thinking turns | Model calls | Prompt tokens | Completion tokens | Reasoning chars | Median wall time per run |
|---|---|---|---|---|---|---|---|---|
| 27B IQ3_S | Never think | 33/40 | 0% | 683 | 32.2M | 262k | 0 | 60 s |
| 27B IQ3_S | **Router @ 0.6** | **37/40** | **19%** | **496** | **17.3M** | **292k** | **352k** | **87 s** |
| 27B IQ3_S | Always think | 40/40 | 100% | 331 | 11.8M | 483k | 1.08M | 141 s |
| 27B Q4_K_M | Never think | 31/40 | 0% | 587 | 20.5M | 204k | 0 | 131 s |
| 27B Q4_K_M | **Router @ 0.6** | **34/40** | **25%** | **447** | **15.2M** | **252k** | **257k** | **141 s** |
| 27B Q4_K_M | Always think | 37/40 | 100% | 363 | 14.6M | 508k | 1.10M | 212 s |

- **Runs took about a third less time.** Paired with always-think, the router's median run took 0.62× as long on IQ3_S
  and 0.67× on Q4_K_M, close to the upstream 0.68×.
- **Thinking cut by about 70%.** Reasoning text dropped by 67% (IQ3_S) and 77% (Q4_K_M) compared with always-think.
  Completion tokens dropped by 40% and 50%.
- **Total tokens didn't go down.** The model took more steps without thinking, so prompt tokens rose by 46% (IQ3_S)
  and 4% (Q4_K_M). Total tokens ended up higher than always-think on both quants (14.7M vs 12.3M on IQ3_S, leaving out
  one pair where always-think timed out; 15.4M vs 15.1M on Q4_K_M). Locally that is cheap thanks to prompt caching. On
  a pay-per-token API it would cancel the savings.
- **Quality stayed close to always-think.** The router never passed a run that always-think failed on the same task
  and rep, and always-think passed three runs that the router failed on each quant. The hidden-test pass rate fell by
  1.3 points on IQ3_S (90% CI −2.7 to −0.1) and 2.8 points on Q4_K_M (90% CI −7.0 to 0.0). The router still beat
  never-think on both quants.
- **The guards did their job.** Under the router, guards forced thinking 17 times on IQ3_S (8 stuck output, 6 loop, 3
  failed command) and 7 times on Q4_K_M (3 stuck output, 4 loop).

**Takeaway:** on a slow local model that thinks at length, the router cuts about a third of the time per run and
most of the thinking, at a small quality cost. It met our speed target on both quants. It missed our pre-set token
target on both, and our quality target (no more than a 5-point loss at the 90% lower bound) on Q4_K_M by 2 points.
If you pay per token rather than in time, always-think is still cheaper.

## 🚀 Quick start

You need Hermes, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU (a CPU works too, see [docs/install.md](docs/install.md)).

**1. Get this repo, Jeff, and the weights**

```sh
git clone https://github.com/chadcox/jeff-hermes && cd jeff-hermes
git clone https://github.com/firelex/jeff vendor/jeff && git -C vendor/jeff checkout 3720e7c   # tested revision
UV="uvx --from uv>=0.12.19 uv"
$UV venv -p 3.12 .venv-gpu && VIRTUAL_ENV=$PWD/.venv-gpu $UV pip install -e 'vendor/jeff[lora,cuda]'
.venv-gpu/bin/hf download mstrasser/jeff-base --revision v1.3 --local-dir models/jeff-base-v1.3
.venv-gpu/bin/hf download mstrasser/jeff-adapter-code-router --revision v1.3 --local-dir models/adapters/code-router
```

**2. Start jeff-serve** (leave it running in its own terminal, or use the systemd unit in [docs/install.md](docs/install.md))

```sh
JEFF_DEVICE=cuda JEFF_CHECKPOINT=models/jeff-base-v1.3 JEFF_ADAPTERS=models/adapters PORT=8765 \
  .venv-gpu/bin/jeff-serve
curl -s 127.0.0.1:8765/health    # in another terminal: {"status": "ready", ...}
```

**3. Install the plugin in Hermes**

Link the plugin into Hermes's plugin folder:

```sh
ln -s "$PWD/plugin/jeff_router" ~/.hermes/plugins/jeff_router
```

Then edit `~/.hermes/config.yaml`. Add `jeff_router` to `plugins.enabled` (keep any plugins already listed there) and
turn the router on:

```yaml
plugins:
  enabled:
    - jeff_router          # plus your existing plugins
  entries:
    jeff_router:
      settings:
        thinking_router: true
```

Without `thinking_router: true` the plugin runs in shadow mode: it asks Jeff and logs the answer, but changes nothing.
Every setting is described in [docs/configuration.md](docs/configuration.md).

**4. Restart Hermes and check it**

Plugins load at startup, so restart Hermes, including the gateway if you run one (for example
`systemctl --user restart hermes-gateway.service`). Run any coding turn, then look at the newest log record:

```sh
tail -n1 ~/.hermes/plugin-data/agent-plugin-jeff_router-*/shadow.jsonl | python3 -m json.tool
```

`router.level` is `"off"` or `"xhigh"` (Jeff's choice), and `main_thinking` says whether the call actually
thought. The full guide, with CPU setup, the systemd unit and exact versions, is **[docs/install.md](docs/install.md)**.

## 🔒 What it touches

When enabled, the plugin:

- **Sends context over HTTP** to jeff-serve at `url` (default `http://127.0.0.1:8765`, local). Each main-turn call
  sends a trimmed view of the conversation, tool names and candidate file paths. Point `url` at a remote host and
  that context leaves your machine.
- **Changes main-model requests.** With `main_thinking: false`, or when the thinking router decides not to think, it
  sets `reasoning_effort: "none"` on main-turn `chat_completions` calls, whichever provider you use.
- **Can write a log** of every call to `shadow.jsonl` in its Hermes plugin-data folder, if you set `log: true`
  (off by default). Records include tool calls, file paths and the router's view of the conversation state, which
  can contain credentials from shell arguments. It rotates above `log_max_mb`.
- **Checks file paths** named in the conversation exist, relative to the session folder. It does not read their
  contents.

It runs no shell commands, starts no background processes beyond a small in-process thread pool, stores no
credentials and sends no telemetry.

## 📚 Documentation

| | |
|---|---|
| [Install](docs/install.md) | jeff-serve on GPU/CPU, weights, systemd, plugin setup |
| [Configuration](docs/configuration.md) | Every setting, provider support, failure behaviour, the log |
| [How it works](docs/how-it-works.md) | Per-call flow, the router question, the guards, code map |
| [Research notes](docs/research/) | Hermes hook API findings, latency and VRAM measurements |
| [Design history](docs/design/) | The v1 shadow-mode design, and why v2 moved to thinking routing |

## 🧪 Development

```sh
uvx --python 3.12 pytest -q      # unit tests; no jeff-serve or Hermes needed
python3 tools/shadow_report.py   # summarise your shadow.jsonl
```

The A/B harness in `eval/` saves each run's session telemetry beside its output, with owner-only permissions.
`python3 eval/report.py OUTDIR` uses those snapshots; older results fall back to the live shadow logs.
A comparison reports **INSUFFICIENT TELEMETRY** when a paired run lacks token counts, has missing or duplicate
call numbers, does not reach a final `stop` reply, or times out. Missing measurements never count as token savings.

## 🙏 Credits

- **[Jeff](https://github.com/firelex/jeff)** and its weights, by Mathias Strasser.
- **[Jeff-Code](https://github.com/firelex/jeff-code)**: the thinking router, router question, and loop/runaway
  guards ported here.
- **[Hermes Agent](https://github.com/NousResearch/hermes-agent)** by Nous Research: the agent this plugs into.

## 📄 License

[MIT](LICENSE)
