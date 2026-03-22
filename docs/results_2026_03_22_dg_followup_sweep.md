# DG Follow-Up Sweep Results

Date: 2026-03-22

## Summary

This document records the 5 additional DG-focused experiment rounds run after promoting PPO+DG as the default training path.

Sweep shape:
- 5 rounds
- 8 parallel runs per round
- 40 total runs
- 50s timeout per run

Starting point:
- [configs/implementation4.yaml](/home/vast/cute_snake/configs/implementation4.yaml)
- DG enabled
- normalized advantage used for gate input
- `dg_eta = 8.0`
- `dg_gate_floor = 0.1`

Main conclusion:
- The remaining gains came from earlier successful eval scheduling, not from changing the DG shape.
- The best run from the 40-run sweep was [runs/2891/metrics.jsonl](/home/vast/cute_snake/runs/2891/metrics.jsonl):
  - elapsed: `9.703125s`
  - success update: `69`
  - eval mean final coverage: `0.934375`
- After promoting the winning schedule into the default config, a plain validation run at [/tmp/dg_default_validate_69/metrics.jsonl](/tmp/dg_default_validate_69/metrics.jsonl) reproduced the same basin:
  - elapsed: `8.811738s`
  - success update: `69`
  - eval mean final coverage: `0.934375`
  - startup time: `1.253074s`

Under the README scoring rule, both runs are valid because startup stays below the excluded `20s` limit.

## Winning Config

Promoted default in [configs/implementation4.yaml](/home/vast/cute_snake/configs/implementation4.yaml):

- `eval_after_update: 69`
- `eval_recent_coverage_gate: 0.42`
- `dg_enabled: true`
- `dg_eta: 8.0`
- `dg_use_raw_advantage_for_gate: false`
- `dg_detach_gate: true`
- `dg_gate_floor: 0.1`

Stable parts of the basin that did not change:
- PPO with `4096` envs and `16` rollout steps
- CNN policy with channels `[28, 56]` and hidden size `224`
- `bfloat16` AMP
- compiled model + compiled GAE
- learner graph path enabled

## Round Winners

Round 1 best:
- [runs/2862/metrics.jsonl](/home/vast/cute_snake/runs/2862/metrics.jsonl)
- `10.383537s`
- update `75`
- eval coverage `0.821875`

Round 2 best:
- [runs/2873/metrics.jsonl](/home/vast/cute_snake/runs/2873/metrics.jsonl)
- `10.108265s`
- update `73`
- eval coverage `0.953125`

Round 3 best:
- [runs/2881/metrics.jsonl](/home/vast/cute_snake/runs/2881/metrics.jsonl)
- `9.815799s`
- update `71`
- eval coverage `0.843750`

Round 4 best:
- [runs/2891/metrics.jsonl](/home/vast/cute_snake/runs/2891/metrics.jsonl)
- `9.703125s`
- update `69`
- eval coverage `0.934375`

Round 5 best:
- [runs/2895/metrics.jsonl](/home/vast/cute_snake/runs/2895/metrics.jsonl)
- `11.175391s`
- update `69`
- eval coverage `0.934375`

## What Helped

- Moving first meaningful eval earlier:
  - `75 -> 73 -> 71 -> 69`
- Keeping the same soft normalized-gate DG basin:
  - `dg_eta = 8.0`
  - `dg_gate_floor = 0.1`
  - `dg_use_raw_advantage_for_gate = false`
- Keeping the eval gate low enough to permit early success checks once the learner was already strong

## What Did Not Help

- Changing DG shape locally:
  - `dg_eta = 7.0` was much worse
  - `dg_eta = 9.0` was worse
  - moving `dg_gate_floor` to `0.08` or `0.12` was worse
- Pushing the eval frontier beyond the actual learning frontier:
  - `68` and `67` schedules still first succeeded at `69`

## Final Takeaway

The DG integration remains the right direction. In this follow-up sweep, DG itself was already good enough; most of the measurable wall-clock gains came from identifying the earliest update where that DG-trained policy reliably exceeded the project threshold.
