from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from snake.cute_kernels import capability_summary
from snake.env_gpu import TorchSnakeBatchEnv
from snake.model import SnakePolicy, load_policy_state
from snake.ppo import _compute_gae, _evaluate_policy_cached, _masked_dist, _sync_perf_counter
from snake.run_dirs import append_jsonl


def build_policy(config: dict[str, Any]) -> SnakePolicy:
    return SnakePolicy(
        board_size=int(config["board_size"]),
        trunk_channels=list(config.get("trunk_channels", [24, 48])),
        hidden_size=int(config["hidden_size"]),
        model_type=str(config.get("model_type", "cnn")),
        transformer_layers=int(config.get("transformer_layers", 4)),
        transformer_heads=int(config.get("transformer_heads", 8)),
        channels_last=bool(config.get("channels_last", False)),
    )


def load_checkpoint_into_policy(model: SnakePolicy, state_dict: dict[str, Any]) -> None:
    load_policy_state(model, state_dict)


def _should_eval(
    update: int,
    *,
    total_updates: int,
    eval_interval: int,
    eval_after_update: int,
    eval_interval_after: int,
    eval_recent_coverage_gate: float,
    recent_coverage: float,
) -> bool:
    if update == total_updates:
        return True
    if eval_recent_coverage_gate > 0.0 and recent_coverage < eval_recent_coverage_gate:
        return False
    if eval_interval_after > 0 and update >= eval_after_update > 0:
        return (update - eval_after_update) % eval_interval_after == 0
    return update % eval_interval == 0
def train(config: dict[str, Any], run_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config["seed"]))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(config.get("allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", True))
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

    model = build_policy(config).to(device)
    if bool(config.get("channels_last", False)) and str(config.get("model_type", "cnn")) == "cnn":
        model = model.to(memory_format=torch.channels_last)
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
    amp_dtype_name = str(config.get("amp_dtype", "float16")).lower()
    amp_dtype = torch.float16 if amp_dtype_name == "float16" else torch.bfloat16

    episode_count_t = torch.zeros((), dtype=torch.long, device=device)
    episode_coverage_sum_t = torch.zeros((), dtype=torch.float32, device=device)
    episode_return_sum_t = torch.zeros((), dtype=torch.float32, device=device)
    episode_length_sum_t = torch.zeros((), dtype=torch.float32, device=device)
    recent_coverage_t = torch.zeros((), dtype=torch.float32, device=device)
    recent_return_t = torch.zeros((), dtype=torch.float32, device=device)
    recent_length_t = torch.zeros((), dtype=torch.float32, device=device)
    have_recent_stats = False
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
                with torch.autocast(device_type=device.type, enabled=scaler_enabled, dtype=amp_dtype):
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

            done_indices = torch.nonzero(dones, as_tuple=False).flatten()
            if done_indices.numel() > 0:
                done_coverages = info["final_coverage"][done_indices]
                done_returns = info["episode_return"][done_indices]
                done_lengths = info["episode_length"][done_indices].to(torch.float32)
                episode_count_t += done_indices.numel()
                episode_coverage_sum_t += done_coverages.sum()
                episode_return_sum_t += done_returns.sum()
                episode_length_sum_t += done_lengths.sum()
                batch_mean_coverage_t = done_coverages.mean()
                batch_mean_return_t = done_returns.mean()
                batch_mean_length_t = done_lengths.mean()
                if not have_recent_stats:
                    recent_coverage_t = batch_mean_coverage_t
                    recent_return_t = batch_mean_return_t
                    recent_length_t = batch_mean_length_t
                    have_recent_stats = True
                else:
                    recent_coverage_t = recent_decay * recent_coverage_t + (1.0 - recent_decay) * batch_mean_coverage_t
                    recent_return_t = recent_decay * recent_return_t + (1.0 - recent_decay) * batch_mean_return_t
                    recent_length_t = recent_decay * recent_length_t + (1.0 - recent_decay) * batch_mean_length_t
                env.reset(dones)
                obs = env.observe()
        if profile_update:
            rollout_seconds = _sync_perf_counter(device) - rollout_start

        if profile_update:
            gae_start = _sync_perf_counter(device)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=scaler_enabled, dtype=amp_dtype):
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
        policy_loss_tensor = torch.zeros((), dtype=torch.float32, device=device)
        value_loss_tensor = torch.zeros((), dtype=torch.float32, device=device)
        entropy_tensor = torch.zeros((), dtype=torch.float32, device=device)

        if profile_update:
            optimize_start = _sync_perf_counter(device)
        for _ in range(int(config["update_epochs"])):
            epoch_indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                stop = start + minibatch_size
                mb_indices = epoch_indices[start:stop]
                mb_obs = b_obs[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_old_logprobs = b_logprobs[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]
                mb_old_values = b_values[mb_indices]
                mb_masks = b_masks[mb_indices]

                with torch.autocast(device_type=device.type, enabled=scaler_enabled, dtype=amp_dtype):
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

                policy_loss_tensor = policy_loss.detach().to(torch.float32)
                value_loss_tensor = value_loss.detach().to(torch.float32)
                entropy_tensor = entropy.detach().to(torch.float32)
        if profile_update:
            optimize_seconds = _sync_perf_counter(device) - optimize_start

        policy_loss_value = float(policy_loss_tensor.item())
        value_loss_value = float(value_loss_tensor.item())
        entropy_value = float(entropy_tensor.item())
        episode_count = int(episode_count_t.item())
        recent_coverage = float(recent_coverage_t.item())
        recent_return = float(recent_return_t.item())
        recent_length = float(recent_length_t.item())
        mean_completed_final_coverage = float((episode_coverage_sum_t / episode_count_t.clamp(min=1)).item())
        mean_completed_episode_return = float((episode_return_sum_t / episode_count_t.clamp(min=1)).item())
        mean_completed_episode_length = float((episode_length_sum_t / episode_count_t.clamp(min=1)).item())

        metrics = {
            "update": update,
            "elapsed_seconds": time.perf_counter() - start_time,
            "update_seconds": time.perf_counter() - update_start,
            "env_steps": update * rollout_steps * num_envs,
            "policy_loss": policy_loss_value,
            "value_loss": value_loss_value,
            "entropy": entropy_value,
            "episodes_completed": episode_count,
            "mean_completed_final_coverage": mean_completed_final_coverage,
            "mean_completed_episode_return": mean_completed_episode_return,
            "mean_completed_episode_length": mean_completed_episode_length,
            "mean_recent_final_coverage": recent_coverage,
            "mean_recent_episode_return": recent_return,
            "mean_recent_episode_length": recent_length,
        }
        if profile_update:
            metrics["rollout_seconds"] = rollout_seconds
            metrics["gae_seconds"] = gae_seconds
            metrics["optimize_seconds"] = optimize_seconds
            metrics["learner_samples_per_second"] = (
                (int(config["update_epochs"]) * batch_size) / max(optimize_seconds, 1e-6)
            )

        if _should_eval(
            update,
            total_updates=int(config["total_updates"]),
            eval_interval=int(config["eval_interval"]),
            eval_after_update=int(config.get("eval_after_update", 0)),
            eval_interval_after=int(config.get("eval_interval_after", 0)),
            eval_recent_coverage_gate=float(config.get("eval_recent_coverage_gate", 0.0)),
            recent_coverage=recent_coverage,
        ):
            if profile_update:
                eval_start = _sync_perf_counter(device)
            evaluation = _evaluate_policy_cached(model, eval_env=eval_env, seed=eval_seed)
            metrics["eval_mean_final_coverage"] = evaluation["mean_final_coverage"]
            metrics["eval_mean_episode_return"] = evaluation["mean_episode_return"]
            with (run_dir / str(config["eval_file"])).open("w", encoding="utf-8") as handle:
                json.dump(evaluation, handle, indent=2)
            if profile_update:
                eval_seconds = _sync_perf_counter(device) - eval_start
                metrics["eval_seconds"] = eval_seconds
            if evaluation["mean_final_coverage"] > best_eval:
                best_eval = evaluation["mean_final_coverage"]
                torch.save({"model": model.state_dict(), "config": config}, checkpoint_dir / "best.pt")
            if bool(config["stop_on_success"]) and evaluation["mean_final_coverage"] >= float(config["success_target"]):
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
        "episodes_completed": int(episode_count_t.item()),
        "success_reached": success_reached,
    }
