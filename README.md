# Cute Snake

Minimal scaffold for a GPU-first Snake PPO prototype using CuTe DSL, `uv`, and `just`.

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
just sweep
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

## Notes

The actual training, evaluation, and visualization scripts are expected to live under `scripts/` and will consume the YAML configs in `configs/`. The packaging layer is intentionally minimal so the implementation can start small and stay GPU-focused.
