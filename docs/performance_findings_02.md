# Performance Findings 02

## Scope

This document records the current optimization state after the first CuTe integration
attempts, trainer/env hot-path cleanup, and multiple 8-GPU wall-clock sweeps.

## Current Best Result

The fastest validated run in the current tree is:

- `runs/1200`
  - mean final coverage over 5 eval episodes: `0.918750`
  - time to first `eval_mean_final_coverage >= 0.80`: `31.033s`
  - success update: `120`
  - env steps at success: `7,864,320`
  - launched from the plain default command after promoting `configs/base.yaml`

This run uses:

- `use_cute_step_core: false`
- `num_envs: 4096`
- `rollout_steps: 16`
- `minibatches: 8`
- `update_epochs: 2`
- `learning_rate: 0.0022`
- `trunk_channels: [24, 48]`
- `hidden_size: 192`
- `amp: true`

This configuration is now written into `configs/base.yaml`.

## Best Known Runs

- `runs/1200`: `0.918750` coverage, `31.033s` to target
- `runs/1100`: `0.918750` coverage, `31.292s` to target
- `runs/1005`: `0.918750` coverage, `31.422s` to target
- `runs/1010`: `0.918750` coverage, `31.564s` to target
- `runs/1016`: `0.875000` coverage, `34.071s` to target

The repeated `4096 x 16`, `epochs=2`, `lr=0.0022`, `[24,48]/192` runs are effectively the
same winner and show the result is stable rather than a one-off.

## Throughput Findings

Observed env-only and training-path measurements in the current tree:

- `scripts/bench_env.py --num-envs 2048 --steps 300 --device cuda`: about `742k` steps/sec before the latest cleanup
- `scripts/bench_env.py --num-envs 2048 --steps 500 --device cuda`: about `769k` steps/sec after the latest cleanup
- winning trainer runs in the `4096 x 16` family typically report about `450k-460k` train-loop SPS

The gap between env-only SPS and trainer SPS is still material. Profiling and run behavior
show the learner path remains the dominant wall-clock cost once the environment is reasonably fast.

## What Helped

The highest-confidence improvements were low-risk hot-path cleanups:

1. Replaced per-forward `F.one_hot(...).to(torch.float32)` board encoding with a frozen identity embedding in `src/snake/model.py`.
2. Switched rollout/eval inference from `torch.no_grad()` to `torch.inference_mode()`.
3. Enabled fused Adam on CUDA in `src/snake/ppo.py`.
4. Removed some per-step env allocations by reusing reward and coverage buffers in `src/snake/env_gpu.py`.
5. Kept the smaller PPO model and fewer optimizer epochs:
   - `[24,48]` trunk instead of `[32,64]`
   - hidden size `192` instead of `256`
   - `update_epochs: 2` instead of `3`

These changes preserved convergence while improving throughput enough to bring the best
validated wall-clock time down into the low-31-second range.

## What Did Not Help

Several plausible ideas did not beat the winning config:

1. `eval_interval: 5`
   - `runs/1101` hit the same success update as the winner, but more frequent eval added overhead.
   - Result: `40.415s`, much worse than `31.033s`.

2. More aggressive learning rates
   - `runs/1011` with `lr=0.0024` topped out below target.
   - `runs/1021` with `lr=0.0023` did succeed, but only at `41.342s`.

3. Larger per-update batches through `4608 x 12`, `5120 x 12`, or `6144 x 8`
   - `runs/1024`, `runs/1025`, and `runs/1016` all succeeded or nearly succeeded.
   - None beat the `4096 x 16` winner in wall-clock time.

4. Smaller model variants
   - `runs/1022` (`hidden=160`) and `runs/1023` (`[20,40]/160`) converged, but slower.

## CuTe Findings

CuTe DSL is installed and working, and the project has a real integrated step-core path.
However, the current CuTe-backed environment path is not the end-to-end winner yet.

Findings:

- Earlier env-only testing showed the CuTe path can improve raw simulator throughput.
- In full training, the current integrated CuTe path still loses on convergence quality and/or total wall clock.
- Recent focused runs:
  - `runs/1014`: best eval coverage `0.718750`
  - `runs/1015`: best eval coverage `0.575000`
  - `runs/1027`: best eval coverage `0.718750`

Interpretation:

- The current fused/split CuTe integration is not preserving learning behavior well enough.
- The project should keep the Torch fast path as default until a more fused CuTe implementation
  matches Torch semantics and beats it on wall clock to target.

## Current Recommendation

Default to the Torch fast path with the winning `4096 x 16` PPO config.

Use the CuTe path only for targeted kernel-development experiments until there is a
Torch-vs-CuTe differential test suite and a CuTe implementation that matches current convergence.
