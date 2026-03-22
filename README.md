# Cute Snake

Minimal GPU-first Snake PPO training repo, trimmed to the current fastest working path.

## Target

The optimization target for this project is:

- Reach `>80%` mean final coverage over the evaluation episodes.
- Minimize wall-clock time to reach that target.
- Observation space, and reward space must stay fixed.

Wall-clock scoring uses this startup rule:

- An initial load/compile step of up to `20s` is tolerated and is not counted toward the run wall clock.
- If the initial load/compile step exceeds `20s`, the run is considered failed.

## Setup

```bash
uv sync --extra dev
```

Install `just` separately if it is not already available on the machine.

## Common Commands

```bash
just setup
just test
just train
just eval RUN_DIR=runs/0001
just visualize RUN_DIR=runs/0001
```

## Run Layout

Every training run should write to a zero-padded incremental directory under `runs/`:

```text
runs/0001/
runs/0002/
runs/0003/
```

Each run directory is expected to contain:

```text
config.yaml
train.log
metrics.jsonl
eval.json
checkpoints/
visualizations/
profiler/
```

The `Justfile` includes a `next-run-dir` helper that creates the next directory and the standard subfolders.

The default training config is [configs/implementation4.yaml](/home/vast/cute_snake/configs/implementation4.yaml).

## Curated Runs

The repo keeps a small archive of representative successful runs:

- `runs/2325`: early fast compile/graph basin
- `runs/2514`: fast no-compile basin
- `runs/2683`: first sub-12s high-margin basin
- `runs/2730`: round-10 sweep winner
- `runs/2737`: current default validation winner

## Notes

The supported code path is the `implementation4` CNN PPO trainer in `src/snake/implementations/implementation4.py`. Older implementation families and broad sweep tooling were removed to keep the tree small.
