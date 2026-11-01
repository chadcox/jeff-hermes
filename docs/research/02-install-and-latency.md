# Step 2: Jeff on CPU, installed and timed

Machine: Ryzen 7 9800X3D (8 cores, 16 threads), 60 GB RAM. It also has an RTX 5080; Jeff was kept off it
(`JEFF_DEVICE=cpu`, `CUDA_VISIBLE_DEVICES=`, CPU-only torch wheel; `nvidia-smi` showed no Jeff process).

## What is installed (all under `~/Desktop/projects/jeff-hermes/`)

| Item | Version | Path |
|---|---|---|
| Jeff source | `firelex/jeff` @ `3720e7c` (package 0.2.0) | `vendor/jeff` |
| Python env | 3.12, torch `2.14.0+cpu`, transformers 5.17.0, peft 0.21.1 | `.venv` |
| Base | `mstrasser/jeff-base` rev `v1.3` (1.7 GB) | `models/jeff-base-v1.3` |
| Adapters | `tools`, `code-router`, `triage`, rev `v1.3` | `models/adapters/<name>` |

Install commands. The local uv is 0.11.16, but Jeff's `pyproject.toml` needs uv ≥ 0.12.19, so a newer uv runs through
`uvx` and the global uv is left as it is. The CPU index avoids downloading the CUDA torch build:

```sh
UV="uvx --from uv>=0.12.19 uv"
$UV venv -p 3.12 .venv
VIRTUAL_ENV=$PWD/.venv $UV pip install -e 'vendor/jeff[lora]' \
  --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/hf download mstrasser/jeff-base --revision v1.3 --local-dir models/jeff-base-v1.3
for n in tools code-router triage; do
  .venv/bin/hf download mstrasser/jeff-adapter-$n --revision v1.3 --local-dir models/adapters/$n
done
```

Serve:

```sh
JEFF_DEVICE=cpu CUDA_VISIBLE_DEVICES= JEFF_CHECKPOINT=models/jeff-base-v1.3 JEFF_ADAPTERS=models/adapters \
  JEFF_QUEUE_MS=2000 PORT=8765 .venv/bin/jeff-serve
```

API: `POST /v1/systemone` with `{"model": "<adapter>" | "jeff", "state": {...}, "questions": {...}}`. The `model`
field is required; `"jeff"` means the base with no adapter. The response gives `answers.<q>.probabilities`,
`choice`, `confidence`, and `usage.input_tokens`. Startup takes about 5 s.

## Latency

Benchmark: `probes/cpu_latency.py`. Requests run one at a time, timed from the client, 30 timed runs after 3
warm-up runs. Three rows (`code-router`, `tools`, `triage`, marked "example") use each adapter's own
`example.json`. The two "hermes-shaped" rows are a short summary built the way the handoff suggests (agent
description, last turn, current message) with Hermes tool names as the options. The rows here are in a different
order from the script's output.

The machine was busy during the run (load average 6–9 from other work), so these numbers include that load.

| Case | Input tokens | Median ms (8 threads) | Median ms (16 threads) | p95 ms (16 threads) |
|---|---:|---:|---:|---:|
| code-router (example) | 201 | 305 | 299 | 304 |
| tools (example) | 363 | 524 | 510 | 547 |
| tools (hermes-shaped) | 349 | 496 | 492 | 530 |
| base, no adapter (hermes-shaped) | 349 | 469 | 485 | 740 |
| triage (example) | 816 | 1542 | 1494 | 1529 |

- **Cost grows with input length, about 1.5–1.8 ms per input token.** A 200–350-token summary takes 0.3–0.5 s.
  An 800-token input takes about 1.5 s. Jeff's README reports 463 ms for about 200 tokens on a 32-thread CPU, which
  is in line with these numbers.
- **16 threads gives no speed-up over the default 8.** Leave the thread count alone.
- **An adapter adds 7–27 ms over the base** at 349 tokens (7 ms at 16 threads, 27 ms at 8). That is within the noise.
- **The first request after a fresh start costs little extra.** On a new server (about 5 s to start), the first
  hermes-shaped request took 554 ms and the next three took 498–502 ms. There is no large one-time warm-up cost.
- An earlier run, made while another adapter was also loaded and the machine was busier, took 30–50% longer
  (`code-router` 456 ms, `tools` 769 ms). Expect some swing under load.

## What this means for the plan

1. **Every model call Jeff passes on to Strata gets slower by 0.3–0.5 s**, for a summary of 350 tokens or fewer.
   A call Jeff answers itself saves a whole Strata round trip, which takes seconds. So Jeff is worth it once it
   answers about 1 in 10 calls or more. That ratio is my estimate, not a measurement; measure it with the real
   Strata latencies.
2. **Keep the summary short.** Cost grows with input length, so a summary of about 300 tokens is the right size.
   The 8K limit is not what restricts it.
3. **The GPU option, measured (see below): about 20 ms per decision.** That makes the time Jeff adds negligible, and
   the input length barely matters. Whether to run it on the 5080 permanently is still the user's call.
4. **The accuracy of a Jeff-built summary is still open.** On the hermes-shaped `tools` request, the user named a
   file path. Jeff chose `search_files` at p = 0.57, while `read_file` was the right call. A threshold around 0.7
   would hand this call to Strata, which is the safe outcome. Picking the threshold needs logged real turns, not
   synthetic examples.
5. **The `tools` adapter picks a tool name, not its arguments.** For the "skip routine steps" plan, the plugin
   still has to fill in the arguments (the path, the pattern) itself. Jeff-Code's `code` adapter has the same
   limitation: its options are read, search, check, and hand over. Check how Jeff-Code fills in arguments before
   building that path. `code`'s card also says it was trained to copy Qwen3.8-27B in a bash-only agent and is not
   meant for other agents or models, so it was not installed.

## GPU measurement (RTX 5080)

A separate venv keeps the CPU `.venv` untouched. The PyPI `torch==2.14.0` is the CUDA 13.0 build and supports sm_120:

```sh
UV="uvx --from uv>=0.12.19 uv"
$UV venv -p 3.12 .venv-gpu
VIRTUAL_ENV=$PWD/.venv-gpu $UV pip install -e 'vendor/jeff[lora,cuda]'
JEFF_DEVICE=cuda JEFF_CHECKPOINT=models/jeff-base-v1.3 JEFF_ADAPTERS=models/adapters \
  JEFF_QUEUE_MS=2000 PORT=8765 .venv-gpu/bin/jeff-serve
```

The optimized `causal_conv1d` kernel is not part of Jeff's `cuda` extra. Without it transformers falls back to a
PyTorch reference implementation and logs a warning. It is built from source against the local CUDA 13.2 toolkit for
sm_120 only, which took about 5.5 minutes:

```sh
export VIRTUAL_ENV=$PWD/.venv-gpu CUDA_HOME=/usr/local/cuda-13.2 PATH=/usr/local/cuda-13.2/bin:$PATH \
  TORCH_CUDA_ARCH_LIST=12.0 CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=8
$UV pip install ninja packaging setuptools wheel
$UV pip install --no-build-isolation causal-conv1d   # 1.7.0
```

### Run 1: idle GPU, without `causal_conv1d`

Same benchmark (`probes/cpu_latency.py 30`). The machine was quiet (load about 1), the card was idle apart from the
desktop, and ComfyUI was not loaded.

| Case | Input tokens | CPU median ms (16 threads) | GPU median ms | GPU p95 ms | Speed-up (approx.) |
|---|---:|---:|---:|---:|---:|
| code-router (example) | 201 | 299 | 20 | 22 | 15× |
| tools (example) | 363 | 510 | 21 | 23 | 24× |
| tools (hermes-shaped) | 349 | 492 | 20 | 23 | 25× |
| base, no adapter (hermes-shaped) | 349 | 485 | 18 | 20 | 27× |
| triage (example) | 816 | 1494 | 31 | 32 | 48× |

- These are client-side times through HTTP, and they match the README's RTX PRO 6000 figures of 22–31 ms. The slow
  laptop report (144–274 ms) does not apply to this card.
- The speed-up column is approximate, not a controlled comparison. The CPU run had a load average of 6–9 from other
  work, while the GPU run had about 1. The GPU medians and p95s stand on their own.
- On GPU the time is mostly a fixed cost. Going from 350 to 816 tokens adds about 10 ms, so the 300-token limit on the
  summary is no longer needed for speed.
- The process shows about 3.0 GB of VRAM in `nvidia-smi` with base plus three adapters, which includes the CUDA
  context and the allocator cache. The README's 1.8 GB figure does not say how it was measured, so the two are not
  directly comparable.
- Decisions match CPU: hermes-shaped `tools` still chose `search_files`, at p = 0.58 on GPU (bf16) versus 0.57 on CPU
  (fp32).

### Run 2: kernel A/B on a shared GPU

This run alternated kernel, fallback, kernel, fallback, starting a fresh server each time with the same benchmark.
Throughout it, Ollama's `llama-server` held 9.7 GB of VRAM and kept the GPU at 98–99% utilization, and the load
average was about 6. The figures are medians in ms, averaged over the two rounds of each setting.

| Case | Input tokens | With kernel | Fallback | Idle GPU, fallback (run 1) |
|---|---:|---:|---:|---:|
| code-router (example) | 201 | 29 | 30 | 20 |
| tools (example) | 363 | 31 | 33.5 | 21 |
| tools (hermes-shaped) | 349 | 31 | 33.5 | 20 |
| base, no adapter (hermes-shaped) | 349 | 30 | 32 | 18 |
| triage (example) | 816 | 43 | 48 | 31 |

- **The kernel saves 1–5 ms (about 3–10%), and the saving grows with input length.** It is installed in `.venv-gpu`
  and worth keeping, but it is not what makes the GPU fast.
- **Sharing the GPU costs about 10–17 ms.** With another model saturating the card, Jeff still answers in 30–48 ms,
  well under the 0.3–0.5 s it takes on CPU.
- With 9.7 GB already taken, Jeff's 2.2–3.0 GB still fit in the 16 GB card, with about 2 GB to spare. A larger
  Ollama or ComfyUI model could leave no room for it.
- The kernel shifts probabilities slightly but does not change any top choice: hermes-shaped `tools` gives p = 0.55
  with the kernel and 0.58 without. That matters only for calls whose p sits right at the routing threshold.
- The kernel's effect was measured only on a shared GPU, where Ollama's load can drift between rounds. Every case was
  faster with the kernel in both rounds, but its gain on an idle GPU is not measured. To measure it, rerun the A/B
  when `nvidia-smi` shows the card idle.

## Coexistence with Hindsight's Ollama

Hindsight runs on this box permanently, so Jeff has to fit around it. Its GPU use comes and goes: the Ollama container
(`hindsight-local-llm-ollama-1`, `ollama/ollama:0.40.9`, `OLLAMA_KEEP_ALIVE=30s`, 32K context) loads a model when a
request arrives and unloads it 30 s after the last one. It serves two models:

- `qwen3:8b` for reflect and mental models. Ollama predicts 9.4 GiB, and `ollama ps` shows 9.8–10.0 GB.
- `gemma4:e2b-it-q4_K_M` for retain and consolidation. Ollama predicts 4.1 GiB.

The two are never loaded at the same time. Ollama evicts one to load the other ("predicted to exceed available
memory, evicting"), so the budget below counts the larger one, `qwen3:8b`. Strata runs on another machine
(`a LAN host`), and the host's own `ollama.service` had nothing loaded.

### Test

Each run starts from a cold Ollama: the harness waits until `ollama ps` in the container is empty. It then creates a
temporary bank, runs a synchronous retain of three facts (loads gemma), runs one reflect (evicts gemma, loads qwen3),
and deletes the bank. Throughout, it samples `nvidia-smi` and `ollama ps` every second. There are three phases:

- **off:** `jeff-serve` not running.
- **idle:** `jeff-serve` running on GPU (`.venv-gpu`, kernel installed, warmed with one request), but sent nothing
  during the run. This isolates Jeff's VRAM reservation.
- **stress:** Jeff also gets a hermes-shaped `tools` request every 0.25 s. That is a stress case, far above real
  traffic of one Jeff call per Hermes model call.

Order: idle, stress, idle, stress, off, off. The box is live, so Hermes's own Hindsight traffic can share the Ollama
queue at any moment.

| Phase | VRAM at start MiB | Retain s (cold load) | Reflect s (model swap) | Placement, each model while loaded | Peak VRAM MiB | Jeff median / p95 / max ms, errors |
|---|---:|---:|---:|---|---:|---|
| off | 2,214 | 2.9 | 11.7 | 100% GPU | 12,108 | — |
| off | 2,214 | 2.9 | 13.2 | 100% GPU | 12,108 | — |
| idle | 4,427 | 2.8 | 9.6 | 100% GPU | 14,176 | — |
| idle | 4,415 | 2.8 | 11.0 | 100% GPU | 14,164 | — |
| stress | 4,415 | 2.8 | 10.4 | 100% GPU | 14,166 | 26 / 37 / 111, 0 of 53 |
| stress | 4,415 | 2.9 | 19.6 | 100% GPU | 14,169 | 24 / 37 / 42, 0 of 87 |

- **Bidirectional coexistence holds for the current models.** With Jeff already holding about 2.2 GB, Ollama
  cold-loaded gemma, then evicted it and loaded qwen3. Each model ran 100% on GPU with no CPU split, and every reflect
  answer was correct. Jeff had no errors through the loads and swaps.
- **Retain is unaffected:** 2.8–2.9 s in every phase.
- **Reflect:** with Jeff off it took 11.7–13.2 s; idle, 9.6–11.0 s; stress, 10.4 s and 19.6 s. Reflect varies run to
  run because it is a tool loop of varying length. Holding the VRAM (idle) costs nothing measurable. The 19.6 s stress
  run is unexplained. It could be GPU contention from four Jeff requests a second, or a longer reflect loop or queued
  Hermes traffic. With two samples it cannot be told apart from noise, and real Jeff traffic is far lighter.
- **Jeff while Ollama loads and generates:** 24–26 ms median, 37 ms p95, max 42–111 ms. An earlier warm-Ollama run
  (since replaced by this one) had a single 552 ms spike. So expect occasional spikes of a few hundred ms during
  Ollama model loads.
- **VRAM budget (16,303 MiB card):** desktop apps about 2.2 GB, Jeff about 2.2 GB, `qwen3:8b` about 9.8 GB, for a
  peak of 14.2 GB and about 2 GB of headroom.
- **The plugin must fail open** (step 1, constraint 8). A short timeout and any error means Strata handles the call.
- **What would break it:**
  - Switching Hindsight's reflect model back to `gpt-oss:20b` (13 GB, still pulled in the container). That plus Jeff
    plus the desktop exceeds 16 GB, so Ollama would split the model onto CPU and reflect would slow sharply.
  - Raising `OLLAMA_CONTEXT_LENGTH` (the KV cache grows), or anything else that wants the card, such as ComfyUI or
    games (the compose file sets keep-alive to 30 s to keep VRAM free for gaming).
  - Jeff holds its 2.2 GB for as long as `jeff-serve` runs; it has no idle unload like Ollama's keep-alive.
