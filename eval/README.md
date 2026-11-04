# eval: A/B harness for the thinking router

This harness compares three ways of running Hermes on the same coding tasks: `off` (the model never thinks), `on` (it always thinks) and `router@<threshold>` (Jeff decides on each call). It records whether the task got solved, how long it took, and the token counts per call taken from the plugin's shadow log.

This is research tooling. Eight small tasks at a few repetitions each point in a direction; they don't support statistical conclusions.

## Tasks

| Task | Kind | What the agent does |
|---|---|---|
| t1 `textstats`, t2 `shop`, t6 `ledger` | fix | Get the failing unit tests passing without editing them |
| t3 `calc`, t4 `slots`, t5 `wrap`, t7 `ranges` | spec | Implement `SPEC.md` and write its own tests |
| t8 `notes` | refactor | Rename a function everywhere it appears |

- `tasks/`: the starting state the agent receives.
- `hidden/`: checks the agent never sees. `grade.py` runs them after the agent finishes.
- `ref/`: reference solutions. They show every hidden check can pass: `python3 eval/grade.py t3 eval/ref/t3`.

## Requirements

- Hermes installed under `~/.hermes`, with the `jeff_router` plugin enabled and `jeff-serve` running.
- A `jeff_router:` settings block and a `model: default:` line in `~/.hermes/config.yaml`.
- Every model you want to compare, served by the provider Hermes is configured to use.

## Running

```sh
python3 eval/run.py set router@0.6 <model>        # backs up config.yaml, then rewrites it
hermes gateway stop --all -y                      # the gateway reads plugin settings only when it starts
python3 eval/run.py run router@0.6 t1 r1 eval/results/mytest
python3 eval/run.py restore                       # puts the original config.yaml back
python3 eval/report.py eval/results/mytest
```

`overnight.sh` runs every model × mode × task combination, repeating until it reaches `MAX_REPS` or the `END` time:
`MODELS="model-a model-b" eval/overnight.sh eval/results/<name>`. It was written for one particular machine, where the gateway runs as the systemd user unit `hermes-gateway.service`, so expect to adapt it to your setup.

## Isolation

- Each run works on a fresh copy of the task in a temp directory with its own git repo.
- The agent runs as your user, though, so it can still reach this repo. While a run is in progress, `run.py` sets `hidden/`, `ref/`, `tasks/` and earlier runs' outputs to mode `000`. An agent that deliberately runs `chmod` can still get past this.
- `tests_touched` in each result flags any run that edited the provided tests.

## Results stay local

`eval/results/` is gitignored, and it should stay that way. The run outputs and the saved telemetry (`*.shadow.jsonl`, mode `600`) contain prompts, file paths and session ids from your machine. Don't commit them. If you want to share numbers, publish `report.md`.
