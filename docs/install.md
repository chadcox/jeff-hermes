# Install

You need three things: a running **jeff-serve** (the Jeff model server), the **`jeff_router` plugin** in Hermes, and
a main model that honours `reasoning_effort: "none"` (llama.cpp, vLLM, and most OpenAI-compatible servers do).

The commands assume the repo is cloned to `~/jeff-hermes`.

## 1. Jeff and its weights

Jeff's `pyproject.toml` needs uv ≥ 0.12.19. If your uv is older, run a newer one through `uvx`:

```sh
cd ~/jeff-hermes
git clone https://github.com/firelex/jeff vendor/jeff     # tested at 3720e7c
UV="uvx --from uv>=0.12.19 uv"
```

### GPU (recommended: about 2.2 GB VRAM, ~70 ms per router question)

```sh
$UV venv -p 3.12 .venv-gpu
VIRTUAL_ENV=$PWD/.venv-gpu $UV pip install -e 'vendor/jeff[lora,cuda]'
```

Optional, but faster: build the `causal_conv1d` kernel. Without it transformers falls back to a slower PyTorch
implementation and logs a warning. Point `CUDA_HOME` at your toolkit and `TORCH_CUDA_ARCH_LIST` at your GPU
(`12.0` is an RTX 50-series card):

```sh
export VIRTUAL_ENV=$PWD/.venv-gpu CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
  TORCH_CUDA_ARCH_LIST=12.0 CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=8
$UV pip install ninja packaging setuptools wheel
$UV pip install --no-build-isolation causal-conv1d
```

### CPU

This works, but each answer takes noticeably longer. See
[research/02-install-and-latency.md](research/02-install-and-latency.md) for the numbers.

```sh
$UV venv -p 3.12 .venv
VIRTUAL_ENV=$PWD/.venv $UV pip install -e 'vendor/jeff[lora]' \
  --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu
```

### Weights

```sh
VENV=.venv-gpu   # or .venv
$VENV/bin/hf download mstrasser/jeff-base --revision v1.3 --local-dir models/jeff-base-v1.3
for n in tools code-router triage; do
  $VENV/bin/hf download mstrasser/jeff-adapter-$n --revision v1.3 --local-dir models/adapters/$n
done
```

The thinking router uses only `code-router`. The `tools` adapter powers the shadow log.

## 2. Run jeff-serve

Run it in the foreground:

```sh
JEFF_DEVICE=cuda JEFF_CHECKPOINT=models/jeff-base-v1.3 JEFF_ADAPTERS=models/adapters \
  JEFF_HOST=127.0.0.1 PORT=8765 JEFF_QUEUE_MS=250 .venv-gpu/bin/jeff-serve
```

Or run it as a systemd user service. The unit warms up the model before it reports started:

```sh
cp deploy/jeff-serve.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now jeff-serve
curl -s 127.0.0.1:8765/health          # {"status": "ready", ...}
```

## 3. Install the plugin

```sh
ln -s ~/jeff-hermes/plugin/jeff_router ~/.hermes/plugins/jeff_router
```

Then enable it in `~/.hermes/config.yaml` and turn on the router:

```yaml
plugins:
  enabled:
    - jeff_router
  entries:
    jeff_router:
      settings:
        thinking_router: true
```

Restart Hermes, including the gateway if you use one. Plugins load at startup.

## 4. Check it

Run a turn, then look at the newest log record:

```sh
tail -n1 ~/.hermes/plugin-data/agent-plugin-jeff_router-*/shadow.jsonl | python3 -m json.tool
```

`router.level` is `"off"` or `"xhigh"`, and `strata_thinking` says whether the call actually thought.

## Tests

The unit tests run on the same Python as Hermes, and they don't need jeff-serve:

```sh
uvx --python ~/.hermes/tools/python-*/bin/python3 --from 'pytest>=8' pytest -q
```

Any Python 3.11+ with pytest also works: `python -m pytest -q`.
