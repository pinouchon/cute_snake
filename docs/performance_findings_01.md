# Performance Findings 01

## Scope

This document records the historical baseline before the later wall-clock sweeps
captured in `docs/performance_findings_02.md`.

## Baseline Outcome

The current project has a working PPO configuration that reaches the target evaluation threshold:

- `runs/0115`: mean final coverage `0.915625` over 5 eval episodes
- `runs/0116`: mean final coverage `0.800000` over 5 eval episodes
- `runs/0118`: mean final coverage `0.871875` over 5 eval episodes using the default command path

The default config at the time of this note is in `configs/base.yaml`.

This is now a historical note only. The current fastest validated configuration is
documented in `docs/performance_findings_02.md`.

## Baseline Throughput

Observed successful training logs show:

- environment/training SPS in the rough range of `70k-75k` environment steps per second near convergence
- success reached at around:
  - `update=100` for `runs/0115`
  - `update=140` for `runs/0118`

One focused environment-only microbenchmark on the original hot path was much worse than the
full trainer SPS made it appear:

- `scripts/bench_env.py --num-envs 2048 --steps 200 --device cuda` measured roughly `3.8k` steps/sec
- `scripts/bench_env.py --num-envs 4096 --steps 200 --device cuda` measured roughly `4.1k` steps/sec

That mismatch pointed directly at reset, done handling, and CPU synchronization costs dominating
the simulator micro-path.

Given:

- `num_envs = 2048`
- `rollout_steps = 32`

That corresponds to:

- `65,536` environment steps per PPO update
- roughly `6.55M` env steps for the fastest successful run from scratch
- roughly `9.18M` env steps for the validated default run

One measurement caveat:

- the very first raw `scripts/bench_env.py` readings in this baseline phase were taken before explicit CUDA synchronization was added to the benchmark script, so those early env-only numbers are only useful as rough directional hints
- training-loop SPS and eval outcomes are the more trustworthy baseline for later comparison

## Likely Bottlenecks Before Optimization

Initial code inspection of the current environment path suggests these hotspots:

1. `src/snake/env_gpu.py`
   The environment is still a Torch/Python implementation, not a compiled CuTe kernel path.

2. `src/snake/env_gpu.py`
   `reset()` is Python-looped over env indices and rebuilds per-env RNG objects.

3. `src/snake/env_gpu.py`
   `_spawn_food()` moves occupancy state to CPU via `.detach().cpu().tolist()` and samples in Python.

4. `src/snake/env_gpu.py`
   Growth handling loops in Python over each growing environment.

5. `src/snake/ppo.py`
   There is currently no fine-grained timing instrumentation for env step, model forward, GAE, optimizer, or eval, so tradeoffs are inferred indirectly from overall SPS.

## Immediate Optimization Direction

Round 1 should do two things before more ambitious changes:

1. Add low-overhead timing instrumentation so env and learner wall time are tracked separately.
2. Remove the obvious Python/CPU overhead in the environment hot path while preserving exact semantics.

## CuTe DSL Direction

CuTe DSL is installed and importable in the project environment. The next implementation goal is to move part of the environment hot path into a CuTe-backed kernel, starting with the most regular batched per-step computations rather than the full simulator rewrite in one jump.

## Historical Status

This document is intentionally left as the pre-optimization baseline. Subsequent CuTe,
throughput, and convergence findings are tracked in `docs/performance_findings_02.md`.
