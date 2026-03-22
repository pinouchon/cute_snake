from __future__ import annotations

import time
from typing import Any

import torch
from torch.distributions import Categorical

from snake.env_gpu import TorchSnakeBatchEnv


def _masked_dist(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    fill_value = torch.finfo(logits.dtype).min
    return Categorical(logits=logits.masked_fill(~mask, fill_value))


def _compute_gae_into(
    *,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    last_advantage = torch.zeros(rewards.shape[1], dtype=torch.float32, device=rewards.device)
    next_values = next_value
    for step in reversed(range(rewards.shape[0])):
        non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * non_terminal * last_advantage
        advantages[step].copy_(last_advantage)
        next_values = values[step]
    returns.copy_(advantages)
    returns.add_(values)
    return advantages, returns


def _sync_perf_counter(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _reset_eval_env(eval_env: TorchSnakeBatchEnv, seed: int) -> torch.Tensor:
    eval_env.done.zero_()
    eval_env.reset_counts.zero_()
    if eval_env.cuda_generator is not None:
        eval_env.cuda_generator.manual_seed(seed)
    return eval_env.reset()


def _evaluate_policy_cached(
    model: torch.nn.Module,
    *,
    eval_env: TorchSnakeBatchEnv,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    obs = _reset_eval_env(eval_env, seed)
    device = obs.device
    episodes = eval_env.num_envs
    finished = torch.zeros(episodes, dtype=torch.bool, device=device)
    final_coverages = torch.zeros(episodes, dtype=torch.float32, device=device)
    final_returns = torch.zeros(episodes, dtype=torch.float32, device=device)
    wins = torch.zeros(episodes, dtype=torch.int32, device=device)

    with torch.no_grad():
        while not torch.all(finished):
            mask = eval_env.action_mask()
            logits, _ = model(obs)
            masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            actions = torch.argmax(masked_logits, dim=-1)
            obs, _, dones, info = eval_env.step(actions)
            just_finished = dones & ~finished
            final_coverages[just_finished] = info["final_coverage"][just_finished]
            final_returns[just_finished] = info["episode_return"][just_finished]
            wins[just_finished] = info["won"][just_finished].to(torch.int32)
            finished |= just_finished

    return {
        "mean_final_coverage": float(final_coverages.mean().item()),
        "mean_episode_return": float(final_returns.mean().item()),
        "coverages": final_coverages.detach().cpu().tolist(),
        "episode_returns": final_returns.detach().cpu().tolist(),
        "wins": int(wins.sum().item()),
    }
