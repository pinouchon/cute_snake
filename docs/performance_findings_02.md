# Performance Findings 02

## Goal

This document records the first two optimization rounds after the baseline in
`docs/performance_findings_01.md`.

The target remained:

- one GPU per run
- maximize wall-clock speed
- stop once mean final coverage over 5 eval episodes reaches `>= 0.80`

## Round 1: Remove Obvious Host Bottlenecks

### Changes

- Fixed the CUDA reset path so batched indexed resets actually write back into the source tensors.
- Removed unconditional Python-side `torch.any(dones)` gating in training, evaluation, and the env benchmark.
- Kept episode stat extraction limited to done subsets instead of copying whole tensors.
- Preserved CPU/reference semantics for food spawning so tests still match the reference env exactly.

### Effect

Environment-only throughput improved immediately:

- before: about `3.8k` env steps/sec at `2048` envs
- after the reset/done cleanup: about `298.7k` env steps/sec at `2048` envs
- same round at `4096` envs: about `597.0k` env steps/sec

This was the largest single simulator speedup.

## Round 2: CuTe Step Core and PPO Sweep

### CuTe Integration

CuTe DSL was kept as a real code path, not just a smoke test:

- `src/snake/cute_kernels.py` now contains a working file-based CuTe compile path
- the env can use a CuTe step core through `use_cute_step_core`
- the CuTe smoke test passes in `tests/test_cute_smoke.py`

### Important Observation

The CuTe step core was not automatically the fastest in an env-only random-action benchmark:

- non-CuTe env-only benchmark at `2048` envs: about `345.1k` steps/sec
- CuTe env-only benchmark at `2048` envs: about `277.5k` steps/sec

But the end-to-end training result was different: the CuTe-enabled training configuration still won on
wall-clock to target once PPO overhead, action sampling, and checkpointing were included.

## Parallel Sweep Results

All 8 GPUs were used for parallel attempts in two rounds.

### First Sweep

Selected successful runs:

- `runs/0301`: baseline-style no-AMP variant, success at update `120`, `64.241s`, eval coverage `0.803125`
- `runs/0302`: `amp=true`, success at update `100`, `46.303s`, eval coverage `0.915625`
- `runs/0303`: `amp=true`, `num_envs=4096`, success at update `110`, `52.863s`, eval coverage `0.884375`
- `runs/0305`: `amp=true`, `num_envs=4096`, `update_epochs=2`, success at update `160`, `56.379s`, eval coverage `0.856250`

Failed or weaker variants showed that larger batches and fewer update passes often raised raw SPS but
hurt sample efficiency enough to lose overall wall-clock.

### Second Sweep

Targeted around the winner from `runs/0302`.

Relevant runs:

- `runs/0318`: `amp=true`, `save_latest=false`, `profile_interval_updates=0`, success at update `100`, `45.507s`, eval coverage `0.915625`
- `runs/0321`: same as `runs/0318` plus `use_cute_step_core=true`, success at update `100`, `41.718s`, eval coverage `0.915625`
- `runs/0322`: `use_cute_step_core=true`, `num_envs=3072`, success at update `110`, `43.825s`, eval coverage `0.931250`

`runs/0321` is the best measured wall-clock result from these rounds.

## Winning Config

The fastest validated configuration from these experiments is:

- `use_cute_step_core: true`
- `amp: true`
- `num_envs: 2048`
- `rollout_steps: 32`
- `minibatches: 8`
- `update_epochs: 4`
- `learning_rate: 1.0e-3`
- `eval_interval: 10`
- `checkpoint_interval: 10`
- `save_latest: false`
- `profile_interval_updates: 0`

Measured result:

- `runs/0321`
- success at update `100`
- `6,553,600` env steps
- `41.718s` elapsed to first eval `>= 0.80`
- first successful eval coverage `0.915625`

## Default Config Promotion

The project default in `configs/base.yaml` was updated to match the winning configuration region:

- CuTe step core enabled
- AMP enabled
- per-update `latest.pt` saving disabled
- profiling disabled by default
- PPO settings restored to the best validated `2048 x 32 x 8 x 4` configuration

## Validation

The final validation steps completed after the default promotion:

- `uv run pytest` -> `6 passed`
- `CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py --run-dir runs/0401`
  - console run hit success at update `100` with eval coverage `0.915625`
- `uv run python scripts/eval.py --run-dir runs/0401`
  - mean final coverage `0.915625`

## Takeaways

1. The biggest win was removing host synchronization and broken batched reset behavior.
2. AMP clearly helped the winner.
3. Larger `num_envs` improved raw SPS but did not beat the best `2048`-env wall-clock result.
4. Disabling per-update checkpoint writes was a real win.
5. The CuTe path is worth keeping, but it should be judged on end-to-end training time, not only an env-only microbenchmark.
