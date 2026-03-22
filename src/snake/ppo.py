from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from snake.cute_kernels import capability_summary
from snake.env_gpu import TorchSnakeBatchEnv
from snake.eval import evaluate_policy
from snake.model import SnakePolicy
from snake.run_dirs import append_jsonl


def _masked_dist(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    fill_value = torch.finfo(logits.dtype).min
    return Categorical(logits=logits.masked_fill(~mask, fill_value))


def _compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.shape[1], dtype=torch.float32, device=rewards.device)
    next_values = next_value
    for step in reversed(range(rewards.shape[0])):
        non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * non_terminal * last_advantage
        advantages[step] = last_advantage
        next_values = values[step]
    returns = advantages + values
    return advantages, returns


def train_ppo(config: dict[str, Any], run_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config["seed"]))

    env = TorchSnakeBatchEnv(
        num_envs=int(config["num_envs"]),
        board_size=int(config["board_size"]),
        max_steps_since_food=int(config["max_steps_since_food"]),
        seed=int(config["seed"]),
        device=device,
        initial_length=int(config["initial_length"]),
        reward_food=float(config["reward_food"]),
        reward_death=float(config["reward_death"]),
        reward_step=float(config["reward_step"]),
    )
    obs = env.reset()

    model = SnakePolicy(
        board_size=int(config["board_size"]),
        hidden_size=int(config["hidden_size"]),
        model_type=str(config["model_type"]),
        transformer_layers=int(config["transformer_layers"]),
        transformer_heads=int(config["transformer_heads"]),
    ).to(device)
    init_checkpoint = config.get("init_checkpoint")
    if init_checkpoint:
        checkpoint = torch.load(str(init_checkpoint), map_location=device)
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    rollout_steps = int(config["rollout_steps"])
    num_envs = int(config["num_envs"])
    minibatches = int(config["minibatches"])
    batch_size = rollout_steps * num_envs
    minibatch_size = batch_size // minibatches

    obs_buf = torch.zeros((rollout_steps, num_envs, env.board_size, env.board_size), dtype=torch.uint8, device=device)
    actions_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    masks_buf = torch.zeros((rollout_steps, num_envs, 4), dtype=torch.bool, device=device)

    best_eval = float("-inf")
    success_reached = False
    metrics_path = run_dir / str(config["metrics_file"])
    checkpoint_dir = run_dir / str(config["checkpoint_dir"])
    scaler_enabled = bool(config.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)

    episode_coverages: list[float] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    start_time = time.perf_counter()
    last_console_log = start_time

    logger.info("CuTe capability: %s", capability_summary())

    for update in range(1, int(config["total_updates"]) + 1):
        update_start = time.perf_counter()
        model.train()

        for step in range(rollout_steps):
            obs_buf[step].copy_(obs)
            mask = env.action_mask()
            masks_buf[step].copy_(mask)

            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=scaler_enabled):
                    logits, values = model(obs)
                dist = _masked_dist(logits, mask)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            actions_buf[step].copy_(actions)
            logprobs_buf[step].copy_(logprobs)
            values_buf[step].copy_(values.to(torch.float32))

            _, rewards, dones, info = env.step(actions)
            rewards_buf[step].copy_(rewards)
            dones_buf[step].copy_(dones.to(torch.float32))

            if torch.any(dones):
                done_indices = torch.nonzero(dones, as_tuple=False).flatten().tolist()
                coverages = info["final_coverage"].detach().cpu().tolist()
                returns = info["episode_return"].detach().cpu().tolist()
                lengths = info["episode_length"].detach().cpu().tolist()
                for index in done_indices:
                    episode_coverages.append(float(coverages[index]))
                    episode_returns.append(float(returns[index]))
                    episode_lengths.append(int(lengths[index]))
                env.reset(dones)

            obs = env.observe()

        with torch.no_grad():
            _, next_values = model(obs)

        advantages, returns = _compute_gae(
            rewards=rewards_buf,
            dones=dones_buf,
            values=values_buf,
            next_value=next_values.to(torch.float32),
            gamma=float(config["gamma"]),
            gae_lambda=float(config["gae_lambda"]),
        )
        advantages = (advantages - advantages.mean()) / (advantages.std().clamp(min=1e-6))

        b_obs = obs_buf.reshape(batch_size, env.board_size, env.board_size)
        b_actions = actions_buf.reshape(batch_size)
        b_logprobs = logprobs_buf.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)
        b_values = values_buf.reshape(batch_size)
        b_masks = masks_buf.reshape(batch_size, 4)

        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0

        for _ in range(int(config["update_epochs"])):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb_indices = indices[start : start + minibatch_size]
                mb_obs = b_obs[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_old_logprobs = b_logprobs[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]
                mb_old_values = b_values[mb_indices]
                mb_masks = b_masks[mb_indices]

                with torch.autocast(device_type=device.type, enabled=scaler_enabled):
                    logits, new_values = model(mb_obs)
                    dist = _masked_dist(logits, mb_masks)
                    new_logprobs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()
                    ratios = (new_logprobs - mb_old_logprobs).exp()
                    unclipped = ratios * mb_advantages
                    clipped = torch.clamp(
                        ratios,
                        1.0 - float(config["clip_coef"]),
                        1.0 + float(config["clip_coef"]),
                    ) * mb_advantages
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    new_values = new_values.to(torch.float32)
                    value_delta = new_values - mb_old_values
                    clipped_values = mb_old_values + value_delta.clamp(
                        -float(config["clip_coef"]),
                        float(config["clip_coef"]),
                    )
                    value_loss_unclipped = (new_values - mb_returns).pow(2)
                    value_loss_clipped = (clipped_values - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    loss = (
                        policy_loss
                        + float(config["value_coef"]) * value_loss
                        - float(config["entropy_coef"]) * entropy
                    )

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()

                policy_loss_value = float(policy_loss.detach().cpu().item())
                value_loss_value = float(value_loss.detach().cpu().item())
                entropy_value = float(entropy.detach().cpu().item())

        metrics = {
            "update": update,
            "elapsed_seconds": time.perf_counter() - start_time,
            "update_seconds": time.perf_counter() - update_start,
            "env_steps": update * rollout_steps * num_envs,
            "policy_loss": policy_loss_value,
            "value_loss": value_loss_value,
            "entropy": entropy_value,
            "episodes_completed": len(episode_coverages),
            "mean_recent_final_coverage": (
                sum(episode_coverages[-100:]) / max(1, len(episode_coverages[-100:]))
            ),
            "mean_recent_episode_return": (
                sum(episode_returns[-100:]) / max(1, len(episode_returns[-100:]))
            ),
            "mean_recent_episode_length": (
                sum(episode_lengths[-100:]) / max(1, len(episode_lengths[-100:]))
            ),
        }

        if update % int(config["eval_interval"]) == 0 or update == int(config["total_updates"]):
            evaluation = evaluate_policy(
                model,
                board_size=int(config["board_size"]),
                max_steps_since_food=int(config["max_steps_since_food"]),
                episodes=int(config["eval_episodes"]),
                seed=int(config["seed"]) + 10_000,
                device=device,
            )
            metrics["eval_mean_final_coverage"] = evaluation["mean_final_coverage"]
            metrics["eval_mean_episode_return"] = evaluation["mean_episode_return"]
            with (run_dir / str(config["eval_file"])).open("w", encoding="utf-8") as handle:
                import json

                json.dump(evaluation, handle, indent=2)
            if evaluation["mean_final_coverage"] > best_eval:
                best_eval = evaluation["mean_final_coverage"]
                torch.save(
                    {"model": model.state_dict(), "config": config},
                    checkpoint_dir / "best.pt",
                )
            if (
                bool(config["stop_on_success"])
                and evaluation["mean_final_coverage"] >= float(config["success_target"])
            ):
                success_reached = True

        if bool(config["save_latest"]) or update % int(config["checkpoint_interval"]) == 0:
            torch.save({"model": model.state_dict(), "config": config}, checkpoint_dir / "latest.pt")
        append_jsonl(metrics_path, metrics)
        now = time.perf_counter()
        if (
            now - last_console_log >= float(config["console_log_interval_seconds"])
            or "eval_mean_final_coverage" in metrics
            or update == int(config["total_updates"])
        ):
            logger.info(
                "update=%s env_steps=%s recent_cov=%.3f recent_return=%.3f eval_cov=%s sps=%.0f",
                update,
                metrics["env_steps"],
                metrics["mean_recent_final_coverage"],
                metrics["mean_recent_episode_return"],
                metrics.get("eval_mean_final_coverage"),
                (rollout_steps * num_envs) / max(1e-6, metrics["update_seconds"]),
            )
            last_console_log = now
        if success_reached:
            logger.info(
                "success_target_reached update=%s eval_cov=%.3f target=%.3f",
                update,
                best_eval,
                float(config["success_target"]),
            )
            break

    return {
        "run_dir": str(run_dir),
        "best_eval_mean_final_coverage": best_eval,
        "episodes_completed": len(episode_coverages),
        "success_reached": success_reached,
    }
