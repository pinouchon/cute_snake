# GPU-First Snake PPO with CuTe DSL

## Scope

This document specifies a research and implementation plan for a reinforcement learning project in which a PPO agent learns to play classic Snake on an `8x8` board. The primary objective is wall-clock speed to a usable policy, not elegance or sample-efficiency in isolation.

The system target is:

- `1x RTX 4090` per training run
- GPU-resident environment stepping and PPO training as much as possible
- CuTe DSL kernels for the environment hot path and other latency-sensitive primitives
- Success criterion: `mean final coverage >= 0.80` over `5` evaluation episodes

Secondary objective:

- Clean experiment infrastructure so the `8` available GPUs can run parallel sweeps later

## Project Tooling

The project should use:

- `uv` for Python environment and dependency management
- `just` via a `Justfile` for common workflows

Minimum expected `just` targets:

- environment setup
- tests
- environment benchmark
- training
- evaluation
- visualization
- sweep launch

The repository should expose at least:

- a training script
- a visualization script

## Fixed Problem Definition

### Environment

- Board size: `8x8`
- Initial snake length: `3`
- Food count: `1`
- Snake grows by `1` when eating food
- Episode ends on:
  - border collision
  - self collision
  - board fill (`length == 64`)
  - starvation timeout, recommended default: `128` steps since the last food

The starvation timeout is the only assumption in this spec that was not explicitly fixed by the request. It is recommended because PPO on Snake otherwise permits very long degenerate episodes that hurt wall-clock performance and make evaluation less meaningful.

### Actions

- Discrete action space with `4` actions: `up/down/left/right`
- Reverse moves are illegal

Recommended contract:

- The policy still outputs `4` logits
- A device-side action mask suppresses the reverse direction before sampling and log-prob computation
- The environment also validates actions and converts any illegal reverse action to `continue straight` as a safety fallback

This keeps the policy interface simple while preserving the game rule.

### Observations

Full-board observation only. No hand-crafted features.

Recommended observation encoding:

- `uint8[64]` flat board
- Cell codes:
  - `0`: empty
  - `1`: food
  - `2`: snake body
  - `3`: head facing up
  - `4`: head facing right
  - `5`: head facing down
  - `6`: head facing left

This satisfies the "full board" requirement while encoding the head direction directly in the board state, which is necessary because reverse moves are illegal.

### Rewards

- `+1.0` for eating food
- `-1.0` for death
- `-0.01` per step
- No additional shaping

### Evaluation

- Default evaluation batch: `5` episodes
- Deterministic policy at eval: masked argmax over action logits
- Metric: `final_coverage = snake_length / 64`
- Success criterion: mean of the 5 episode final coverage values is `>= 0.80`

## Primary Optimization Goal

The main metric is:

- `minutes to first passing evaluation`

Supporting metrics:

- environment steps per second
- PPO updates per second
- end-to-end wall-clock to each evaluation checkpoint
- best and latest eval mean coverage
- compile/warmup overhead

Sample-efficiency matters only insofar as it reduces time-to-threshold.

## Design Principles

The design should borrow the useful parts of PufferLib's performance model without copying its stack:

- in-place buffer updates
- contiguous rollout storage
- zero-copy or near-zero-copy interfaces between env and learner
- minimal Python in the hot path
- batch-first design from the start
- custom kernels where common RL code becomes a bottleneck

PufferLib explicitly attributes its speed to in-place operations, shared memory buffers, zero-copy batching, and minimizing copying in vectorized environments. It also notes that custom kernels matter enough that it ships a GPU advantage kernel for faster training. That is directly relevant here because Snake is a tiny environment where launch overhead and buffer movement can dominate compute.

## System Architecture

### High-Level Stack

- CuTe DSL:
  - environment reset kernel
  - environment step kernel
  - food spawn kernel logic
  - action-mask kernel
  - optional GAE/return kernel
  - optional masked sampling kernel
- PyTorch:
  - policy/value network
  - optimizer
  - PPO loss
  - AMP / mixed precision
- Runtime:
  - all rollout buffers on GPU
  - persistent compiled kernels
  - optional CUDA Graph capture once tensor shapes are frozen

### Why CuTe DSL Fits

Per NVIDIA's CuTe DSL documentation, CuTe DSL is a Python DSL for dynamic compilation of high-performance GPU kernels with `@jit` and `@kernel`, DLPack integration with frameworks like PyTorch, and JIT caching for repeated calls. The current docs also expose AOT compilation and TVM FFI support for lower host overhead and faster PyTorch interop.

For this project, the value is not tensor-core-heavy math inside the environment. The value is:

- low-level control for branchy GPU env logic
- direct GPU kernels without writing a C++ extension first
- persistent caching or AOT to remove repeated compile cost
- direct framework interop for `torch.Tensor` buffers

### GPU State Layout

Use a structure-of-arrays layout for the simulator state:

- `head_idx[N] : uint8`
- `tail_idx[N] : uint8`
- `length[N] : uint8`
- `heading[N] : uint8`
- `food_idx[N] : uint8`
- `alive[N] : uint8`
- `steps_since_food[N] : uint16`
- `episode_step[N] : uint16`
- `rng_state[N] : uint64`
- `occupancy[N] : uint64`
- `body_ring[N, 64] : uint8`
- `ring_head[N] : uint8`
- `ring_tail[N] : uint8`
- `board[N, 64] : uint8`

Notes:

- `8x8` fits exactly in a `uint64` occupancy bitboard.
- `board[N,64]` exists for direct policy consumption.
- `occupancy[N]` exists for fast collision tests and empty-cell enumeration.
- The board is updated incrementally in place each step rather than rebuilt from scratch.

### Environment Step Semantics

One environment step should be handled by one CUDA thread initially. This is the preferred v1 mapping because each step touches only a few cells and contains branchy game logic.

Per-step algorithm:

1. Read masked action and current heading.
2. Compute candidate next head cell.
3. Check wall collision.
4. Check self collision using `occupancy`, with the standard Snake exception:
   moving into the current tail cell is legal if the snake is not growing on this step because the tail vacates.
5. If alive:
   - move head
   - update ring buffer
   - update occupancy bitboard
   - if food eaten:
     - increment length
     - reset starvation counter
     - spawn a new food uniformly from empty cells
   - else:
     - pop tail
6. Update `board[N,64]` in place:
   - old head becomes body
   - new head code reflects direction
   - old tail cell becomes empty if no growth
   - food cell is set
7. Emit reward, done flag, truncation flag, and final coverage.
8. Reset finished envs immediately in-kernel or via a follow-up reset kernel.

### Food Spawning

Food spawning must be exact and uniform over empty cells.

Recommended method:

- Compute `empty_mask = ~occupancy & valid_board_mask`
- Compute `k = rng % popcount(empty_mask)`
- Select the `k`-th set bit from `empty_mask`

This is fast because the board is only `64` cells.

### RNG

Use a per-environment GPU RNG state with a simple counter-based or small-state generator. The main requirements are:

- deterministic under seed control
- cheap per step
- usable for both action sampling and food spawning

Reproducibility matters, but cryptographic quality does not.

## PPO Training Design

### Recommended Initial Network

The model should be small enough that env throughput does not starve on model latency, but strong enough to learn long-horizon pathing without feature engineering.

Recommended v1 policy:

- Input: `board[N,64]`
- Convert to one-hot or learned embedding of `7` cell codes on GPU
- Reshape to `N x C x 8 x 8`
- Shared trunk:
  - `Conv3x3(32)`
  - `Conv3x3(64)`
  - `Flatten`
  - `Linear(128)`
- Actor head: `Linear(4)`
- Critic head: `Linear(1)`

Use a shared trunk unless experiments show that separate towers materially improve time-to-threshold.

### Precision

- Environment state remains integer-typed
- Policy forward/backward uses AMP
- Prefer `bf16` on 4090 if stable in the chosen stack, else `fp16`
- Value targets, returns, and advantage normalization remain `fp32`

### PPO Contract

Recommended starting defaults:

- `gamma = 0.99`
- `gae_lambda = 0.95`
- `clip_coef = 0.2`
- `value_coef = 0.5`
- `entropy_coef = 0.01`
- `max_grad_norm = 0.5`
- `adam_lr = 3e-4`
- `update_epochs = 2-4`

These are starting points, not fixed design constraints. Since the objective is wall-clock time, the first sweep should focus on whichever settings reach the threshold fastest, not whichever settings look most standard.

### Rollout Buffer

All rollout tensors should be allocated up front on GPU in contiguous `[T, N, ...]` form:

- obs
- actions
- logprobs
- rewards
- dones
- values
- action masks

Derived tensors:

- advantages
- returns

The buffer is append-only during rollout and reused in place across PPO iterations.

### Recommended First Search Region

Start with a small but meaningful sweep over:

- `num_envs`: `16384`, `32768`, `65536`
- `rollout_len`: `16`, `32`, `64`
- `minibatches`: `4`, `8`
- `learning_rate`: `1e-4`, `3e-4`, `1e-3`
- `gamma`: `0.99`, `0.995`
- `entropy_coef`: `0.0`, `0.01`, `0.02`

Why this region:

- Too few environments wastes the GPU
- Too many environments increase PPO staleness and rollout memory
- Too long rollouts improve throughput but can hurt time-to-threshold
- Snake is small enough that these tradeoffs need to be measured, not assumed

## Kernel and Runtime Strategy

### Minimize Host Overhead

Because this environment is tiny, host overhead is a first-order problem.

Recommended strategy:

1. Keep all simulator buffers as persistent GPU tensors.
2. Pre-compile kernels explicitly with `cute.compile`.
3. Persist cache with `CUTE_DSL_CACHE_DIR`.
4. For benchmark runs, move to AOT export once kernel signatures are stable.
5. Prefer TVM FFI or the fastest supported CuTe-to-PyTorch interop path if it lowers launch overhead for `torch.Tensor` arguments.

This follows the CuTe DSL docs directly:

- JIT caching exists and can reduce repeat compile overhead.
- `cute.compile` returns reusable executors.
- AOT exists specifically to remove JIT overhead in production-style runs.
- TVM FFI exists for lower runtime overhead and better framework interop.

### CUDA Graphs

After shapes are fixed, capture the repeated rollout step sequence with CUDA Graphs if the PyTorch and CuTe call boundaries cooperate cleanly.

Candidate captured sequence:

- policy forward
- action masking
- action sampling
- env step
- value writeback

If graph capture becomes fragile, keep it out of v1. Correctness and low Python overhead matter more than forcing graphs everywhere.

### Advantage/Return Computation

GAE is a likely hotspot because it is simple, repeated, and easy to fuse.

Plan:

- v1: implement in PyTorch on GPU
- v2: replace with CuTe DSL kernel if profiler shows meaningful wall time

This mirrors PufferLib's general lesson that common RL plumbing can deserve custom kernels when training speed is the goal.

## Benchmarking Plan

### Stage 0: Semantic Oracle

Build a tiny reference Snake implementation used only for correctness testing:

- exact reset semantics
- exact reward logic
- exact food spawning behavior
- exact action-mask semantics
- deterministic seeded episodes

This should not be the fast path. It exists to validate the GPU simulator.

### Stage 1: Environment Microbenchmarks

Measure, on one 4090:

- reset throughput
- step throughput
- throughput as a function of `N`
- step latency with and without writing full observations
- cost of immediate reset on done

Goal:

- find the regime where GPU occupancy is high and host overhead is not dominant

### Stage 2: PPO End-to-End Profiling

For each candidate config, record:

- env SPS
- learner samples/sec
- percent wall time in:
  - env stepping
  - model forward
  - backward/optimizer
  - GAE/returns
  - eval
  - compile/warmup

Goal:

- identify the true bottleneck before optimizing further

### Stage 3: Time-to-Threshold Sweeps

Run parallel sweeps across the 8 GPUs. Rank experiments by:

1. time to first `>= 0.80` mean final coverage on 5 eval episodes
2. best coverage reached within the allowed run time
3. end-to-end stability across seeds

### Stage 4: Post-Threshold Hardening

Once the threshold is reached:

- freeze a reference config
- rerun on multiple seeds
- compare JIT cache vs AOT
- compare PyTorch GAE vs CuTe GAE
- lock evaluation semantics

## Recommended Initial Milestones

### Milestone 1

Correct GPU simulator with matching CPU/reference semantics.

Exit criteria:

- deterministic seeded tests pass
- illegal reverse moves handled correctly
- food spawning uniformity sanity checks pass
- full-board observation updates correctly

### Milestone 2

Single-GPU PPO training loop working end to end.

Exit criteria:

- no host-device sync in the hot loop except deliberate logging/eval boundaries
- stable loss values
- non-trivial learning above random baseline

### Milestone 3

Profiling-informed optimization.

Exit criteria:

- top 2 wall-clock bottlenecks identified with profiler traces
- at least one bottleneck removed by kernel/runtime changes

### Milestone 4

Threshold run.

Exit criteria:

- mean final coverage `>= 0.80` over `5` eval episodes
- result reproducible on at least `3` seeds before calling it solved

The original request only fixed `5` eval episodes by default. The stronger `3`-seed requirement is a recommended internal standard, not a change to the success criterion.

## Risks

### Risk 1: Launch Overhead Dominates

This is the most likely failure mode for a tiny environment.

Mitigations:

- persistent device buffers
- explicit CuTe compilation and cache reuse
- AOT once stable
- larger `num_envs`
- optional CUDA Graph capture

### Risk 2: PPO Instability from Oversized Batches

Very large environment counts can maximize SPS while hurting policy improvement speed.

Mitigations:

- sweep `num_envs` and `rollout_len` together
- track time-to-threshold, not SPS alone

### Risk 3: Observation Encoding Too Weak or Too Expensive

A flat `uint8[64]` board is compact, but the network still needs direction-aware spatial reasoning.

Mitigations:

- start with board-coded head direction
- compare one-hot vs learned embedding
- avoid larger architectures unless profiling proves they help wall-clock

### Risk 4: Simulator Correctness Bugs Masquerade as RL Failure

Snake edge cases are easy to get wrong:

- moving into tail legality
- food respawn on nearly full boards
- reset semantics after terminal transitions

Mitigations:

- reference implementation
- property tests
- long deterministic trajectory comparisons

## Recommended Repository Shape

```text
docs/
  snake_rl_spec.md
Justfile
pyproject.toml
uv.lock
src/
  snake/
    env_reference.py
    env_gpu.py
    cute_kernels.py
    model.py
    ppo.py
    eval.py
    config.py
scripts/
  bench_env.py
  train.py
  eval.py
  visualize.py
  sweep.py
configs/
  base.yaml
  sweep_wallclock.yaml
runs/
  0001/
    config.yaml
    train.log
    metrics.jsonl
    eval.json
    checkpoints/
    visualizations/
    profiler/
  0002/
    ...
artifacts/
  cute_cache/
tests/
  test_env_semantics.py
  test_food_spawn.py
  test_gpu_matches_reference.py
```

## Run Directory Structure

Every launched run should create a new incrementing output directory under `runs/`:

- `runs/0001/`
- `runs/0002/`
- `runs/0003/`

Recommended contents for each run directory:

- `config.yaml`: fully resolved config for reproducibility
- `train.log`: human-readable training log
- `metrics.jsonl`: append-only structured training metrics
- `eval.json`: latest and best evaluation summaries
- `checkpoints/`: saved model checkpoints
- `visualizations/`: rendered episodes, plots, or videos
- `profiler/`: optional profiler traces for benchmark runs

The run id allocator should be simple and deterministic:

- scan `runs/`
- choose the next zero-padded integer width `4`
- create the directory before training starts

Training, evaluation, and visualization scripts should all accept an explicit run directory override, but default to this incremental scheme.

## Immediate Research Plan

### Phase A: De-Risk Semantics

- implement a tiny reference Snake
- lock the exact transition rules
- decide and document starvation timeout

### Phase B: Build the Fast Path

- implement CuTe DSL reset and step kernels
- keep observation/state buffers fully resident on GPU
- integrate directly with PyTorch tensors

### Phase C: Make PPO Cheap

- keep the network small
- keep rollout storage contiguous
- keep GAE and loss on GPU
- eliminate Python from the per-step loop as much as possible

### Phase D: Optimize by Profile, Not Taste

- benchmark env-only first
- then profile full PPO
- only write extra kernels where traces justify it

### Phase E: Sweep for Time-to-Threshold

- use all 8 GPUs for independent sweeps
- sort runs by minutes to pass the evaluation threshold
- keep the fastest reproducible configuration

## Sources

- NVIDIA CuTe DSL introduction:
  - https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html
- NVIDIA CuTe DSL control flow:
  - https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html
- NVIDIA CuTe DSL JIT caching:
  - https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html
- NVIDIA CuTe DSL AOT:
  - https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.html
- NVIDIA CuTe DSL framework integration:
  - https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/framework_integration.html
- PufferLib documentation:
  - https://puffer.ai/docs.html
- PufferLib blog:
  - https://puffer.ai/blog.html
