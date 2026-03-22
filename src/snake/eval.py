from __future__ import annotations

from typing import Any

import torch

from snake.env_gpu import TorchSnakeBatchEnv


def evaluate_policy(
    model: torch.nn.Module,
    *,
    board_size: int,
    max_steps_since_food: int,
    episodes: int,
    seed: int,
    device: torch.device,
    use_cute_step_core: bool = False,
) -> dict[str, Any]:
    model.eval()
    env = TorchSnakeBatchEnv(
        num_envs=episodes,
        board_size=board_size,
        max_steps_since_food=max_steps_since_food,
        seed=seed,
        device=device,
        initial_length=3,
        reward_food=1.0,
        reward_death=-1.0,
        reward_step=-0.01,
        use_cute_step_core=use_cute_step_core,
    )
    obs = env.reset()
    finished = torch.zeros(episodes, dtype=torch.bool, device=device)
    final_coverages = torch.zeros(episodes, dtype=torch.float32, device=device)
    final_returns = torch.zeros(episodes, dtype=torch.float32, device=device)
    wins = torch.zeros(episodes, dtype=torch.int32, device=device)

    while not torch.all(finished):
        mask = env.action_mask()
        with torch.no_grad():
            logits, _ = model(obs)
            masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            action = torch.argmax(masked_logits, dim=-1)
        obs, _, dones, info = env.step(action)
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
