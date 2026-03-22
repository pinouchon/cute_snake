# Cute Snake Project Summary

Last updated from the local working tree on `2026-03-22` UTC.

This document is intended to export the current goal, implementation state, experiment history, and open questions of the repository for external analysis. It combines:

- the formal project spec in `docs/snake_rl_spec.md`
- historical findings in `docs/performance_findings_01.md` and `docs/performance_findings_02.md`
- the current code in `src/`, `scripts/`, `configs/`, and `tests/`
- the run archive under `runs/`
- the current local working tree, including uncommitted changes

## Executive Summary

- The project goal is to train a PPO agent to solve `8x8` Snake as fast as possible in wall-clock time on a single `RTX 4090`, with success defined as mean final coverage `>= 0.80` over `5` deterministic evaluation episodes.
- The environment, rollout storage, and trainer are GPU-first, but the implementation is still primarily PyTorch/Torch tensor code rather than a fully device-owned CuTe simulator.
- The repository has a working from-scratch baseline. Many runs clear the target.
- The best fully documented baseline in the historical docs is `runs/1200`, which reaches the threshold in `31.033s`.
- The current run archive contains newer scratch runs not reflected in the older docs:
  - `runs/1912` (`implementation1`) reaches the target in `28.666s` with final eval coverage `0.91875`.
  - `runs/2032` (`implementation4`) reaches the target in `26.518s`, currently the fastest scratch threshold crossing in the archive, but with weaker final eval quality (`0.80625`) than the strongest `implementation1`/`implementation4` quality runs.
- CuTe integration exists and works, but the winning scratch runs still use `use_cute_step_core: false`.
- The newest working tree adds multiple implementation labels, more model variants, and new sweep configs, but some of the written docs are stale relative to the live code.

## Repository Status

### Git / Working Tree Status

- Git history is shallow: `first version`, `cleanup`, `round 1`, `round 2`.
- The local working tree is dirty and includes uncommitted tracked changes plus untracked files.
- The summary below therefore reflects the live workspace, not only committed history.

Important uncommitted additions present locally:

- implementation dispatch via `src/snake/api.py`
- `src/snake/implementations/`
- `configs/implementation2.yaml`
- `configs/implementation3.yaml`
- `configs/implementation4.yaml`
- `configs/sweep_implementation4_round1.yaml`
- `configs/sweep_implementation4_round2.yaml`
- `configs/sweep_implementation4_round3.yaml`
- `docs/implementations.md`
- `tests/test_model.py`

Important tracked files with local modifications:

- `configs/base.yaml`
- `docs/performance_findings_02.md`
- `scripts/train.py`
- `scripts/eval.py`
- `scripts/visualize.py`
- `scripts/launch_sweep.py`
- `src/snake/config.py`
- `src/snake/env_gpu.py`
- `src/snake/model.py`
- `src/snake/ppo.py`
- `tests/test_env_semantics.py`

### Validation Status

Current local test suite result:

- `uv run pytest`
- Result: `9 passed in 2.69s`

The current tests cover:

- run directory allocation
- reference-environment semantics
- one fixed-action reference-vs-batch-env parity check
- action-mask semantics
- model forward shapes for axial and implementation4 policy construction
- optional CuTe smoke test when CuTe and CUDA are available

The tests do not provide a comprehensive Torch-vs-CuTe differential suite for full training behavior.

## Project Goal and Formal Spec

The formal project spec is in `docs/snake_rl_spec.md`.

### Primary Objective

- Train a Snake agent on an `8x8` board using PPO.
- Optimize for `minutes to first passing evaluation`, not for sample efficiency in isolation.
- Target hardware per run: `1x RTX 4090`.
- Longer-term goal: infrastructure should scale across `8` GPUs for sweeps.

### Success Criterion

- Mean final coverage `>= 0.80`
- Measured over `5` deterministic evaluation episodes
- Evaluation policy uses masked argmax

### Fixed Environment Semantics

- Board size: `8x8`
- Initial snake length: `3`
- One food at a time
- Growth: `+1` on food
- Episode ends on:
  - wall collision
  - self collision
  - full board
  - starvation timeout, default `128` steps since last food

### Observation / Action / Reward Contract

- Observation is the full board.
- The encoded cell values are:
  - `0` empty
  - `1` food
  - `2` snake body
  - `3` head up
  - `4` head right
  - `5` head down
  - `6` head left
- Action space is `4` actions: up / right / down / left.
- Reverse moves are illegal.
- Policy still outputs `4` logits.
- Reverse actions are masked and also sanitized in the environment as "continue straight".
- Rewards:
  - `+1.0` food
  - `-1.0` death
  - `-0.01` step penalty

### Intended System Direction from the Spec

The spec aims for:

- GPU-resident rollout buffers
- CuTe kernels for env hot paths
- optional CuTe kernels for GAE or other RL plumbing
- minimal host overhead
- explicit measurement of wall-clock bottlenecks

That target is only partially realized in the current implementation.

## What Is Actually Implemented

### Runtime Surface

Main entry points:

- `scripts/train.py`
- `scripts/eval.py`
- `scripts/visualize.py`
- `scripts/launch_sweep.py`
- `scripts/summarize_sweep.py`
- `scripts/bench_env.py`

Workflow tooling:

- `uv` for environment/package management
- `just` targets for setup, tests, env benchmark, training, eval, visualization, and a simple sweep entrypoint

Run layout:

- Each run is saved under `runs/NNNN/`
- Standard subdirectories:
  - `checkpoints/`
  - `visualizations/`
  - `profiler/`
- Standard files:
  - `config.yaml`
  - `train.log`
  - `metrics.jsonl`
  - `eval.json`

### Configuration

The normalized config path is in `src/snake/config.py`.

Important current characteristics:

- nested config files under `env`, `model`, `ppo`, `runtime`, `logging`
- normalization flattens those keys for training code
- implementation selection exists through `config["implementation"]`
- there are also dormant DQN-style config fields in `normalize_config`, but there is no DQN implementation in the current tree

### Environment

Primary environment implementation:

- `src/snake/env_gpu.py`
- class: `TorchSnakeBatchEnv`

Reference/oracle environment:

- `src/snake/env_reference.py`
- class: `ReferenceSnakeEnv`

Current environment design in code:

- batched Torch tensor environment
- state stored in tensors such as:
  - `board`
  - `occupancy`
  - `body`
  - `start`
  - `length`
  - `heading`
  - `food`
  - `done`
  - `episode_step`
  - `steps_since_food`
  - `episode_return`
- action mask is precomputed via a `4x4` lookup table indexed by current heading
- resets and stepping are GPU-aware
- partial CuTe-assisted stepping exists, but the environment is not yet a fully fused CuTe simulator

Notable gap versus the spec:

- The spec recommends a compact bitboard-oriented GPU state layout with more complete device-side ownership.
- The live environment still uses a higher-level Torch tensor layout and retains host/PyTorch book-keeping around the step path.

### CuTe Integration

Relevant files:

- `src/snake/cute_kernels.py`
- `src/snake/cute_fused_experiment.py`
- `scripts/cute_fused_experiment.py`
- `scripts/cute_smoke.py`
- `tests/test_cute_smoke.py`

Current CuTe state:

- CuTe import is optional.
- A smoke kernel exists and is tested conditionally.
- A step-core kernel exists.
- A larger fused step-update experiment exists.
- The env can enable CuTe step assistance with `use_cute_step_core: true`.
- The default winning configs still keep this disabled.

Important practical conclusion:

- CuTe is integrated enough to support kernel experimentation.
- CuTe is not yet the default or fastest reliable end-to-end training path.

### Policy Model

Policy implementation:

- `src/snake/model.py`
- class: `SnakePolicy`

Current supported model types:

- `cnn`
- `mlp`
- `transformer`
- `axial`

Current notable model details:

- board cell codes are embedded with a frozen identity `Embedding(7, 7)` rather than building one-hot tensors per forward
- CNN trunk depth is now variable rather than fixed to exactly two conv layers
- axial attention support exists for research experiments
- policy outputs:
  - actor logits over `4` actions
  - scalar critic value

This is more flexible than the original narrower baseline model.

### Training Code

Main PPO trainer:

- `src/snake/ppo.py`

Implementation-specific alternative trainer:

- `src/snake/implementations/implementation4.py`

Shared API dispatch:

- `src/snake/api.py`
- `src/snake/implementations/__init__.py`

Common trainer features:

- rollout buffers are preallocated on device
- masked categorical policy
- GAE on GPU in PyTorch
- AMP support
- fused Adam enabled on CUDA
- evaluation env is cached and reused during training
- metrics written as JSONL

### Important Truth About "Implementations"

The written docs and the live code are not fully aligned.

Live code reality:

- `implementation1` builds a policy and calls `train_ppo`
- `implementation2` currently builds the same policy shape as `implementation1` and also calls `train_ppo`
- `implementation3` currently only changes default model-type defaults and still calls `train_ppo`
- `implementation4` is the only implementation in the current tree with its own separate training loop

This means:

- `implementation2` is not currently a distinct trainer algorithm in code, despite `docs/implementations.md` describing it that way
- `implementation3` exists in code now, but is not the future placeholder described by the older doc
- `implementation4` is the actual new trainer stack under active experimentation

### What `implementation4` Actually Changes

`implementation4` is still PPO, not a different RL family.

The main practical difference versus `src/snake/ppo.py` is its minibatch data movement strategy:

- it shuffles full flattened epoch tensors once per epoch
- it then slices contiguous minibatch views from those shuffled tensors

This differs from the baseline trainer, which repeatedly indexes minibatches out of the base flattened buffers.

Likely consequence:

- better memory locality / lower indexing overhead in the learner step
- same broad PPO objective and env semantics

## Evaluation and Artifact Semantics

### Evaluation

Evaluation path:

- `src/snake/eval.py`
- reused during training through `_evaluate_policy_cached` in `src/snake/ppo.py`

Evaluation characteristics:

- deterministic masked argmax
- final outputs include:
  - mean final coverage
  - mean episode return
  - per-episode coverages
  - per-episode returns
  - wins

### Visualization

Visualization path:

- `scripts/visualize.py`

Characteristics:

- loads a checkpoint
- uses the reference CPU environment
- rolls out the policy with masked argmax
- prints an ASCII board animation

This is mainly a debugging/inspection tool rather than part of the high-performance path.

## Run Archive Overview

Current run archive summary from `runs/`:

- `270` completed runs with both `config.yaml` and `metrics.jsonl`
- `18` warm-start runs with `init_checkpoint`
- `17` of those warm-start runs reached the target
- `100` scratch runs reached the target

High-level implementation counts:

| Implementation | Run Count | Scratch/Warm Mix | Success Count | Best Time To Target |
| --- | ---: | --- | ---: | ---: |
| `implementation1` | 160 | mostly scratch, plus all warm-start family | 89 | `3.174s` if warm-start is included, `28.666s` best scratch |
| `implementation2` | 40 | scratch | 5 | `31.260s` |
| `implementation3` | 21 | scratch | 1 | `47.832s` |
| `implementation4` | 49 | scratch | 22 | `26.518s` |

Important warning:

- Naive run ranking is misleading unless warm-start runs are filtered out.
- The very fastest runs in the archive are not valid from-scratch comparisons.

## Experiment History and Findings

### 1. Historical Baseline Phase

Sources:

- `docs/performance_findings_01.md`

Key historical results:

- `runs/0115`: mean final coverage `0.915625`
- `runs/0116`: mean final coverage `0.800000`
- `runs/0118`: mean final coverage `0.871875`

Historical trainer performance noted in the doc:

- roughly `70k-75k` trainer SPS near convergence
- success around update `100` to `140`

Important historical bottlenecks identified:

- Python-looped reset logic
- CPU food spawning path
- Python growth handling
- lack of fine-grained timing instrumentation

This phase established that the project could already solve the task, but left large questions about where wall-clock time was going.

### 2. Environment Throughput / CuTe Cleanup Phase

Sources:

- `docs/performance_findings_02.md`

Documented results from that phase:

- env-only benchmark improved from roughly `3.8k-4.1k` steps/sec in an earlier baseline microbenchmark to roughly `742k-769k` steps/sec after cleanup
- trainer SPS in the winning family reached roughly `450k-460k`
- learner cost remained material after env cleanup

Documented improvements that helped:

- frozen identity embedding instead of per-forward one-hot generation
- `torch.inference_mode()` in inference paths
- fused Adam on CUDA
- some buffer reuse in the environment
- smaller model and fewer PPO epochs

Documented best run in that note:

- `runs/1200`
- coverage `0.918750`
- `31.033s` to target
- success update `120`

Historical conclusion from that note:

- Torch fast path should remain default
- CuTe path should stay experimental until convergence matches Torch

### 3. Warm-Start Continuation Family

The archive contains a fast family of runs that are not valid from-scratch comparisons.

Key fact:

- `runs/0117`, `1301`, `1310` through `1327` all load `init_checkpoint: runs/0105/checkpoints/best.pt`

Examples:

- `runs/1320`: `3.220s` to target
- `runs/1323`: `3.174s` to target
- `runs/1327`: `3.648s` to target, with `use_cute_step_core: true`

Interpretation:

- These are continuation/warm-start experiments.
- They are useful for measuring adaptation speed and short-horizon follow-up behavior.
- They should not be used when comparing true scratch convergence.

This distinction matters because some scripts or naive run summaries will otherwise rank them as the "best" runs.

### 4. Scratch-Only Aggressive Follow-Up

This is documented in the newer appended section of `docs/performance_findings_02.md`.

Key result:

- The original documented scratch winner in that note was `runs/1030` at `30.453s`.
- `runs/1500` was a clean control rerun at `31.271s`.

Aggressive scratch variants that failed to improve wall-clock:

- more PPO epochs
- more aggressive entropy
- low-horizon variants
- weak-critic / unclipped-value variants
- throughput-first CuTe attempts

Key interpretation recorded in the doc:

- the limiting factor appears to be learning dynamics, not raw env speed alone
- aggressively shortening horizon hurts more than it helps
- sub-10-second convergence from scratch was not achieved

### 5. Newer Archive Result: Baseline Improved Beyond the Older Docs

The live archive contains at least one newer scratch `implementation1` run that beats the older documented `runs/1200` / `runs/1030` baseline.

Most important run:

- `runs/1912`
- `implementation1`
- scratch
- `4096 x 16`, `update_epochs=2`, `lr=0.0022`, `[24,48]` trunk, `hidden=192`, `amp=true`
- `28.666s` to target
- final eval mean final coverage `0.91875`

Interpretation:

- the historical docs are directionally correct about the winning baseline family
- but they are no longer numerically up to date with the best run present in the archive

### 6. Implementation2 Experiments

Observed run family:

- major runs in the `1600`-`1818` range

Important nuance:

- `implementation2` is currently not a genuinely distinct trainer in code
- it calls the same `train_ppo` function as `implementation1`

Empirical findings anyway:

- early `1600` and `1700` runs were poor, often topping out near `0.05` to `0.42` coverage
- later runs using the same strong `4096 x 16`, `epochs=2`, `lr=0.0022` family succeeded

Best implementation2 run:

- `runs/1811`
- `31.260s` to target
- final coverage `0.91875`

Other successful implementation2 runs:

- `runs/1813`: `39.193s`, final coverage `0.875`
- `runs/1814`: `37.762s`, final coverage `0.821875`
- `runs/1816`: `42.815s`, final coverage `0.803125`
- `runs/1818`: `41.802s`, final coverage `0.88125`

Interpretation:

- These results should not be read as strong evidence for a distinct "implementation2 algorithm".
- In the live code, they are closer to relabeled baseline PPO runs than to a separate trainer stack.

### 7. Implementation3 / Architecture Experiments

Observed run family:

- major runs in the `1900`-`1928` range

What was tested:

- axial model variants
- some CNN overrides under the `implementation3` label
- some MLP variants

Results:

- most axial runs were weak, often around `0.05` to `0.52` best coverage
- best axial-like result in this family appears to be `runs/1921` at `0.51875` best coverage, still far from target

Only successful implementation3 run:

- `runs/1926`
- `47.832s` to target
- final coverage `0.853125`
- crucially, `model_type: cnn`, not `axial`

Interpretation:

- The archive does not currently support the claim that axial attention is the best next model direction for this task.
- The only successful run under the `implementation3` label used a CNN, which again reflects the naming/code-structure mismatch.

### 8. Implementation4 Round 1-3 Experiments

Observed run family:

- `2001` through `2055`
- configured by:
  - `configs/implementation4.yaml`
  - `configs/sweep_implementation4_round1.yaml`
  - `configs/sweep_implementation4_round2.yaml`
  - `configs/sweep_implementation4_round3.yaml`

This is the most important newer experiment track not fully captured in the older docs.

#### Early implementation4 failures

The earliest low-learning-rate implementation4 runs mostly failed:

- `2001`-`2013` were mostly near-random or very weak
- this includes many runs with `lr` in the `0.0001`-`0.00025` range

Interpretation:

- the new trainer stack needed a much hotter learning-rate region than those early initial probes

#### Implementation4 matching the strong baseline family

Implementation4 can match the strong baseline-quality result when run with the same general hyperparameter family as the old winner:

- `runs/2024`: `30.523s`, final coverage `0.91875`
- `runs/2018`: `31.396s`, final coverage `0.91875`

Interpretation:

- implementation4 is not merely a low-quality speed hack
- it can recover the old quality level with the `lr=0.0022` family

#### Fastest scratch threshold crossings in the entire archive

Current fastest scratch runs:

- `runs/2032`: `26.518s`, final coverage `0.80625`
- `runs/2048`: `26.682s`, final coverage `0.80625`
- `runs/2040`: `26.950s`, final coverage `0.80625`
- `runs/2036`: `27.980s`, final coverage `0.80000`
- `runs/2055`: `28.816s`, final coverage `0.80000`

Shared config pattern:

- `implementation4`
- `num_envs: 4096`
- `rollout_steps: 16`
- `update_epochs: 2`
- `learning_rate: 0.0028`
- `use_cute_step_core: false`
- seed `106`

Interpretation:

- This is currently the fastest scratch family for crossing the official threshold.
- It appears to trade quality margin for earlier threshold arrival.

#### Higher-quality implementation4 variants

Implementation4 also has stronger-quality runs that are still competitive on wall clock:

- `runs/2039`: `30.314s`, final coverage `0.853125`
- `runs/2053`: `33.589s`, final coverage `0.912500`
- `runs/2024`: `30.523s`, final coverage `0.918750`

This suggests a Pareto shape:

- `lr=0.0028` gives the fastest threshold crossings but often only modestly clears the bar
- `lr=0.0022` or some nearby settings produce slower but much stronger final policies

#### Larger-batch implementation4 variants

The newer implementation4 sweeps also tested:

- `8192 x 8`
- `16384 x 4`

Observed result:

- these larger-batch / shorter-horizon variants generally regressed versus `4096 x 16`
- examples:
  - `runs/2041`: best coverage `0.76875`
  - `runs/2042`: best coverage `0.66875`
  - `runs/2044`: best coverage `0.528125`

Interpretation:

- the old lesson still applies: raw throughput-oriented larger batches can hurt early learning enough to lose on time-to-threshold

#### Limitations of the implementation4 evidence

Important caveat:

- the successful implementation4 family is almost entirely repeated on seed `106`

So the evidence supports:

- repeatability across multiple reruns of the same seed/config

But it does not yet strongly support:

- multi-seed robustness

## CuTe-Specific Findings Across the Archive

High-confidence conclusions:

- CuTe is installed/usable in the intended environment and the smoke path works when available.
- CuTe-assisted step logic exists in the env.
- Full scratch winners still use `use_cute_step_core: false`.
- The archive contains CuTe runs with good results, but those do not currently displace the Torch fast path as the safest recommendation.

Examples:

- warm-start `runs/1327` with `use_cute_step_core: true` is fast, but warm-start evidence is not enough
- scratch `runs/1014`, `1015`, and `1027` under older CuTe experiments were weak
- recent implementation4 winners continue to prefer `use_cute_step_core: false`

External interpretation:

- CuTe work should currently be treated as enabling infrastructure and a research branch, not the production/default training path

## Current Best Known Recommendations

### If the objective is "safest strong scratch baseline"

Use the documented `implementation1` family:

- `4096` envs
- `16` rollout steps
- `8` minibatches
- `2` update epochs
- `lr=0.0022`
- `[24,48]` CNN trunk
- hidden size `192`
- AMP on
- `use_cute_step_core: false`

Evidence:

- `runs/1912`: `28.666s`, final coverage `0.91875`
- `runs/1030`: `30.453s`, final coverage `0.91875`
- `runs/1200`: `31.033s`, final coverage `0.91875`

### If the objective is "fastest threshold crossing seen so far"

Use the newer `implementation4` speed-first family:

- `4096` envs
- `16` rollout steps
- `8` minibatches
- `2` update epochs
- `lr=0.0028`
- same small CNN
- AMP on
- `use_cute_step_core: false`

Evidence:

- `runs/2032`: `26.518s`
- `runs/2048`: `26.682s`
- `runs/2040`: `26.950s`

Main tradeoff:

- faster to first `>= 0.80`
- often lower final eval quality than the best baseline-quality runs

## Main Open Problems

### 1. Documentation Drift

Current docs are partially stale relative to the live code and archive:

- `docs/performance_findings_02.md` does not cover the implementation4 `20xx` sweep family
- `docs/implementations.md` does not match the current code reality for implementations `2` and `3`

### 2. Multi-Seed Validation

The newest fastest implementation4 family has repeated same-seed success, not broad seed coverage.

Needed:

- rerun `implementation4` winner families across multiple seeds
- compare not only threshold time but final quality and failure rate

### 3. CuTe Differential Correctness / Convergence

Still missing:

- a stronger Torch-vs-CuTe differential validation suite
- clear evidence that CuTe preserves learning behavior in end-to-end training

### 4. Profiling Coverage on Latest Runs

The latest strong configs generally run with:

- `profile_interval_updates: 0`

So the archive is rich in outcome data but thin in fresh per-stage timing breakdowns.

### 5. Algorithm/Codebase Clarity

The repository now contains:

- implementation dispatch
- multiple labels
- dormant DQN config keys

But not all labels correspond to distinct algorithms yet.

For outside analysis, this means:

- names alone are not a trustworthy proxy for actual training logic
- the code paths need to be checked directly

## Key Files for External Review

### Project-Level Docs

- `README.md`
- `docs/snake_rl_spec.md`
- `docs/performance_findings_01.md`
- `docs/performance_findings_02.md`
- `docs/implementations.md`

### Core Runtime

- `src/snake/config.py`
- `src/snake/api.py`
- `src/snake/run_dirs.py`
- `src/snake/env_gpu.py`
- `src/snake/env_reference.py`
- `src/snake/model.py`
- `src/snake/ppo.py`
- `src/snake/implementations/implementation1.py`
- `src/snake/implementations/implementation2.py`
- `src/snake/implementations/implementation3.py`
- `src/snake/implementations/implementation4.py`

### CuTe Experimentation

- `src/snake/cute_kernels.py`
- `src/snake/cute_fused_experiment.py`
- `scripts/cute_fused_experiment.py`
- `scripts/cute_smoke.py`

### Sweep / Analysis Tooling

- `scripts/train.py`
- `scripts/eval.py`
- `scripts/visualize.py`
- `scripts/launch_sweep.py`
- `scripts/summarize_sweep.py`
- `configs/sweep_wallclock.yaml`
- `configs/sweep_implementation4_round1.yaml`
- `configs/sweep_implementation4_round2.yaml`
- `configs/sweep_implementation4_round3.yaml`

### Representative Runs To Inspect

Historical baseline:

- `runs/0115`
- `runs/0118`

Documented old winner family:

- `runs/1030`
- `runs/1100`
- `runs/1200`

Warm-start continuation family:

- `runs/1320`
- `runs/1323`
- `runs/1327`

Aggressive scratch failures:

- `runs/1501`
- `runs/1505`
- `runs/1517`

Implementation2 / 3 evidence:

- `runs/1811`
- `runs/1921`
- `runs/1926`

Newest implementation4 speed/quality frontier:

- `runs/2024`
- `runs/2032`
- `runs/2039`
- `runs/2053`

## Bottom-Line State

The project is no longer at the "can Snake PPO work at all?" stage. That question is settled.

The current state is:

- there is a working GPU-first PPO Snake trainer
- there is a valid scratch baseline with many successful runs
- CuTe integration exists but is still experimental
- the main optimization question has shifted from "get it working" to "how to lower wall-clock further without losing learning quality"

The most important current frontier is the tradeoff between:

- `implementation1` / baseline-quality scratch runs that reach about `0.91875` final coverage in about `28.7-31.0s`
- `implementation4` speed-first runs that hit the threshold in about `26.5-27.0s` but often with much smaller margin above the target

If external analysis is meant to recommend the next step, the highest-leverage unresolved questions appear to be:

- whether the `implementation4` speedup survives proper multi-seed validation
- whether its speed comes from a learner-path improvement that can be kept while recovering the stronger final quality regime
- whether CuTe should be pushed further into a truly fused env path, or whether the learner is now the more important bottleneck
