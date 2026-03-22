from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from snake.cute_kernels import capability_summary
from snake.env_gpu import TorchSnakeBatchEnv
from snake.model import SnakePolicy, load_policy_state
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


def train_ppo(config: dict[str, Any], run_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config["seed"]))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        precision = config.get("matmul_precision")
        if precision:
            torch.set_float32_matmul_precision(str(precision))

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
        use_cute_step_core=bool(config.get("use_cute_step_core", False)),
    )
    obs = env.reset()
    eval_seed = int(config["seed"]) + 10_000
    eval_env = TorchSnakeBatchEnv(
        num_envs=int(config["eval_episodes"]),
        board_size=int(config["board_size"]),
        max_steps_since_food=int(config["max_steps_since_food"]),
        seed=eval_seed,
        device=device,
        initial_length=int(config["initial_length"]),
        reward_food=float(config["reward_food"]),
        reward_death=float(config["reward_death"]),
        reward_step=float(config["reward_step"]),
        use_cute_step_core=bool(config.get("use_cute_step_core", False)),
    )

    model = SnakePolicy(
        board_size=int(config["board_size"]),
        trunk_channels=list(config.get("trunk_channels", [32, 64])),
        hidden_size=int(config["hidden_size"]),
        model_type=str(config["model_type"]),
        transformer_layers=int(config["transformer_layers"]),
        transformer_heads=int(config["transformer_heads"]),
    ).to(device)
    init_checkpoint = config.get("init_checkpoint")
    if init_checkpoint:
        checkpoint = torch.load(str(init_checkpoint), map_location=device)
        load_policy_state(model, checkpoint["model"])
    if bool(config.get("compile_model", False)) and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
    optimizer_kwargs: dict[str, Any] = {"lr": float(config["learning_rate"])}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

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

    episode_count = 0
    episode_coverage_sum = 0.0
    episode_return_sum = 0.0
    episode_length_sum = 0.0
    recent_coverage = 0.0
    recent_return = 0.0
    recent_length = 0.0
    recent_decay = 0.9
    start_time = time.perf_counter()
    last_console_log = start_time

    logger.info("CuTe capability: %s", capability_summary())

    for update in range(1, int(config["total_updates"]) + 1):
        update_start = time.perf_counter()
        profile_update = int(config["profile_interval_updates"]) > 0 and (
            update % int(config["profile_interval_updates"]) == 0 or update == 1
        )
        rollout_seconds = 0.0
        gae_seconds = 0.0
        optimize_seconds = 0.0
        eval_seconds = 0.0
        checkpoint_seconds = 0.0
        model.train()

        if profile_update:
            rollout_start = _sync_perf_counter(device)
        for step in range(rollout_steps):
            obs_buf[step].copy_(obs)
            mask = env.action_mask()
            masks_buf[step].copy_(mask)

            with torch.inference_mode():
                with torch.autocast(device_type=device.type, enabled=scaler_enabled):
                    logits, values = model(obs)
                dist = _masked_dist(logits, mask)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            actions_buf[step].copy_(actions)
            logprobs_buf[step].copy_(logprobs)
            values_buf[step].copy_(values.to(torch.float32))

            obs, rewards, dones, info = env.step(actions)
            rewards_buf[step].copy_(rewards)
            dones_buf[step].copy_(dones.to(torch.float32))

            done_count = int(dones.sum().item())
            if done_count:
                done_coverages = info["final_coverage"][dones]
                done_returns = info["episode_return"][dones]
                done_lengths = info["episode_length"][dones]
                episode_count += done_count
                episode_coverage_sum += float(done_coverages.sum().item())
                episode_return_sum += float(done_returns.sum().item())
                episode_length_sum += float(done_lengths.sum().item())
                batch_mean_coverage = float(done_coverages.mean().item())
                batch_mean_return = float(done_returns.mean().item())
                batch_mean_length = float(done_lengths.to(torch.float32).mean().item())
                if episode_count == done_count:
                    recent_coverage = batch_mean_coverage
                    recent_return = batch_mean_return
                    recent_length = batch_mean_length
                else:
                    recent_coverage = recent_decay * recent_coverage + (1.0 - recent_decay) * batch_mean_coverage
                    recent_return = recent_decay * recent_return + (1.0 - recent_decay) * batch_mean_return
                    recent_length = recent_decay * recent_length + (1.0 - recent_decay) * batch_mean_length
                env.reset(dones)
                obs = env.observe()
        if profile_update:
            rollout_seconds = _sync_perf_counter(device) - rollout_start

        if profile_update:
            gae_start = _sync_perf_counter(device)
        with torch.inference_mode():
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
        if profile_update:
            gae_seconds = _sync_perf_counter(device) - gae_start

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

        if profile_update:
            optimize_start = _sync_perf_counter(device)
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
                    value_loss_unclipped = (new_values - mb_returns).pow(2)
                    if bool(config.get("use_value_clipping", True)):
                        value_delta = new_values - mb_old_values
                        clipped_values = mb_old_values + value_delta.clamp(
                            -float(config["clip_coef"]),
                            float(config["clip_coef"]),
                        )
                        value_loss_clipped = (clipped_values - mb_returns).pow(2)
                        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    else:
                        value_loss = 0.5 * value_loss_unclipped.mean()
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
        if profile_update:
            optimize_seconds = _sync_perf_counter(device) - optimize_start

        metrics = {
            "update": update,
            "elapsed_seconds": time.perf_counter() - start_time,
            "update_seconds": time.perf_counter() - update_start,
            "env_steps": update * rollout_steps * num_envs,
            "policy_loss": policy_loss_value,
            "value_loss": value_loss_value,
            "entropy": entropy_value,
            "episodes_completed": episode_count,
            "mean_completed_final_coverage": episode_coverage_sum / max(1, episode_count),
            "mean_completed_episode_return": episode_return_sum / max(1, episode_count),
            "mean_completed_episode_length": episode_length_sum / max(1, episode_count),
            "mean_recent_final_coverage": recent_coverage,
            "mean_recent_episode_return": recent_return,
            "mean_recent_episode_length": recent_length,
        }
        if profile_update:
            metrics["rollout_seconds"] = rollout_seconds
            metrics["gae_seconds"] = gae_seconds
            metrics["optimize_seconds"] = optimize_seconds

        if update % int(config["eval_interval"]) == 0 or update == int(config["total_updates"]):
            if profile_update:
                eval_start = _sync_perf_counter(device)
            evaluation = _evaluate_policy_cached(
                model,
                eval_env=eval_env,
                seed=eval_seed,
            )
            metrics["eval_mean_final_coverage"] = evaluation["mean_final_coverage"]
            metrics["eval_mean_episode_return"] = evaluation["mean_episode_return"]
            with (run_dir / str(config["eval_file"])).open("w", encoding="utf-8") as handle:
                json.dump(evaluation, handle, indent=2)
            if profile_update:
                eval_seconds = _sync_perf_counter(device) - eval_start
                metrics["eval_seconds"] = eval_seconds
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

        if profile_update and (bool(config["save_latest"]) or update % int(config["checkpoint_interval"]) == 0):
            checkpoint_start = _sync_perf_counter(device)
        if bool(config["save_latest"]) or update % int(config["checkpoint_interval"]) == 0:
            torch.save({"model": model.state_dict(), "config": config}, checkpoint_dir / "latest.pt")
        if profile_update and (bool(config["save_latest"]) or update % int(config["checkpoint_interval"]) == 0):
            checkpoint_seconds = _sync_perf_counter(device) - checkpoint_start
            metrics["checkpoint_seconds"] = checkpoint_seconds
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
        "episodes_completed": episode_count,
        "success_reached": success_reached,
    }
