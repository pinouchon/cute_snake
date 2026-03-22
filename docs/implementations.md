# Implementations

## Overview

The codebase now supports multiple training implementations behind a shared script/API surface.

- `implementation1`
  - the current working baseline
  - hybrid Torch GPU env
  - minibatched PPO trainer
  - fastest validated scratch path so far
- `implementation2`
  - experimental fast-update path
  - same observation and reward spaces
  - full-batch PPO-style updates instead of shuffled minibatches
  - intended to reduce optimizer overhead and make alternative training dynamics easy to test

The scripts:

- `scripts/train.py`
- `scripts/eval.py`
- `scripts/visualize.py`

select the implementation through `config["implementation"]`.

## Current Status

### Implementation1

Backed by:

- `src/snake/implementations/implementation1.py`
- `src/snake/ppo.py`
- `src/snake/env_gpu.py`
- `src/snake/model.py`

This remains the reference implementation and the fastest verified from-scratch solution.

### Implementation2

Backed by:

- `src/snake/implementations/implementation2.py`

This is a new trainer stack that:

- keeps the same env semantics and policy interface
- removes minibatch shuffling and random permutation overhead
- applies PPO-style updates over the full rollout batch directly

Current experiments show that it is much faster in raw SPS, but much worse in learning quality than
`implementation1`.

## API Surface

Shared implementation dispatch lives in:

- `src/snake/api.py`
- `src/snake/implementations/__init__.py`

The shared contract is:

- `build_policy(config)`
- `load_checkpoint_into_policy(model, state_dict)`
- `train(config, run_dir, logger)`

## Next Likely Direction

The current evidence suggests that `implementation2` as a trainer-only change is not enough.

The most promising next implementation is likely:

- a more fully fused CuTe env path that owns the entire reset/step/update pipeline device-side
- while keeping the working PPO-style learning dynamics from `implementation1`

That would be a future `implementation3`, or a major revision of `implementation2`.
