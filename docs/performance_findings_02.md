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

## Scratch-Only Aggressive Follow-Up

After the warm-start experiments were ruled out as invalid for the real target, I ran
additional scratch-only sweeps anchored to the `1030/1020/1200` family.

Key outcome:

- The best scratch run is still `runs/1030` at `30.453s` to first `>= 0.80` eval coverage.
- A clean control rerun, `runs/1500`, matched that family at `31.271s`.

Aggressive scratch variants that did **not** beat the baseline:

- `runs/1505`
  - `4096 x 16`, `update_epochs=3`, `entropy_coef=0.01`, `use_value_clipping=false`
  - best aggressive variant
  - reached target in `45.104s`
- `runs/1503`
  - `2048 x 32`, `update_epochs=4`, `lr=0.001`, `entropy_coef=0.01`
  - reached target in `60.987s`
- `runs/1504`
  - `3072 x 16`, `update_epochs=4`, `lr=0.0012`, `entropy_coef=0.01`
  - reached target in `56.792s`
- `runs/1502`
  - `2048 x 16`, `update_epochs=6`, `lr=0.0015`, `entropy_coef=0.01`
  - reached target in `66.539s`

High-variance low-horizon scratch variants all regressed:

- `runs/1501`
  - `4096 x 8`, `update_epochs=4`, smaller model
  - failed, best eval coverage `0.71875`
- `runs/1510`
  - aggressive low-horizon PPO with shorter discount horizon
  - failed, best eval coverage `0.625`
- `runs/1511`
  - `8192 x 8`, larger PPO step, smaller model
  - failed, best eval coverage `0.50625`
- `runs/1512`
  - `4096 x 12`, more aggressive PPO settings
  - failed, best eval coverage `0.56875`
- `runs/1513`
  - `4096 x 8`, `update_epochs=5`, larger model
  - failed, best eval coverage `0.484375`
- `runs/1514`
  - weak-critic / unclipped-value aggressive PPO
  - failed, best eval coverage `0.521875`
- `runs/1515`
  - loose clipping, no value clipping, `4096 x 12`
  - failed, best eval coverage `0.503125`
- `runs/1516`
  - slower high-quality-update variant
  - failed, best eval coverage `0.484375`
- `runs/1517`
  - throughput-first CuTe path, `8192 x 4`, `update_epochs=1`
  - failed, best eval coverage `0.165625`

Interpretation:

- From-scratch PPO appears to be limited more by learning dynamics than raw env speed.
- Lowering horizon aggressively hurts early learning more than it helps update freshness.
- The current CuTe path can drive extreme env SPS, but that does not translate into earlier
  success from scratch under the current PPO setup.
- The bounded code changes added for experimentation:
  - `mlp` model type
  - variable-depth CNN trunks
  - `use_value_clipping` toggle
  did not unlock a meaningful wall-clock win in the first scratch sweeps.

Current conclusion:

- Sub-10-second convergence from scratch was **not** achieved.
- The next realistic path is a larger change than hyperparameter tuning:
  1. a truly fused end-to-end CuTe transition kernel that removes the remaining host-side env patching, or
  2. a materially different training update path that cuts optimizer cost while preserving sample efficiency.
