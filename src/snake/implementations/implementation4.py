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
from snake.ppo import _compute_gae_into, _evaluate_policy_cached, _masked_dist, _sync_perf_counter
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


def _is_eval_scheduled(
    update: int,
    *,
    total_updates: int,
    eval_interval: int,
    eval_after_update: int,
    eval_interval_after: int,
) -> bool:
    if update == total_updates:
        return True
    if eval_interval_after > 0 and update >= eval_after_update > 0:
        return (update - eval_after_update) % eval_interval_after == 0
    return update % eval_interval == 0


def _maybe_compile_model(model: SnakePolicy, config: dict[str, Any]) -> SnakePolicy:
    if not bool(config.get("compile_model", False)) or not hasattr(torch, "compile"):
        return model
    compile_mode = str(config.get("compile_mode", "reduce-overhead"))
    options: dict[str, Any] | None = None
    if bool(config.get("compile_disable_cudagraphs", False)):
        options = {"triton.cudagraphs": False}
    if options is not None:
        return torch.compile(model, options=options)  # type: ignore[return-value]
    return torch.compile(model, mode=compile_mode)  # type: ignore[return-value]


def _maybe_compile_gae(config: dict[str, Any]):
    if not bool(config.get("compile_gae", False)) or not hasattr(torch, "compile"):
        return _compute_gae_into
    options: dict[str, Any] | None = None
    if bool(config.get("compile_disable_cudagraphs", False)):
        options = {"triton.cudagraphs": False}
    if options is not None:
        return torch.compile(_compute_gae_into, options=options)
    return torch.compile(_compute_gae_into, mode=str(config.get("compile_mode", "reduce-overhead")))


def _sample_masked_actions(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    fill_value = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, fill_value)
    probs = torch.softmax(masked_logits, dim=-1)
    actions = torch.multinomial(probs, 1, replacement=True, generator=generator).squeeze(-1)
    logprobs = torch.log_softmax(masked_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
    return actions, logprobs


def _index_select_into(dst: torch.Tensor, src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    torch.index_select(src, 0, indices, out=dst)
    return dst


class _MinibatchWorkspace:
    def __init__(
        self,
        *,
        device: torch.device,
        board_size: int,
        minibatch_size: int,
    ) -> None:
        self.obs = torch.empty((minibatch_size, board_size, board_size), dtype=torch.uint8, device=device)
        self.actions = torch.empty(minibatch_size, dtype=torch.long, device=device)
        self.old_logprobs = torch.empty(minibatch_size, dtype=torch.float32, device=device)
        self.advantages = torch.empty(minibatch_size, dtype=torch.float32, device=device)
        self.returns = torch.empty(minibatch_size, dtype=torch.float32, device=device)
        self.old_values = torch.empty(minibatch_size, dtype=torch.float32, device=device)
        self.masks = torch.empty((minibatch_size, 4), dtype=torch.bool, device=device)

    def load(
        self,
        *,
        indices: torch.Tensor,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        masks: torch.Tensor,
    ) -> None:
        _index_select_into(self.obs, obs, indices)
        _index_select_into(self.actions, actions, indices)
        _index_select_into(self.old_logprobs, old_logprobs, indices)
        _index_select_into(self.advantages, advantages, indices)
        _index_select_into(self.returns, returns, indices)
        _index_select_into(self.old_values, old_values, indices)
        _index_select_into(self.masks, masks, indices)


class _PermutationBank:
    def __init__(
        self,
        *,
        batch_size: int,
        minibatches: int,
        minibatch_size: int,
        update_epochs: int,
        bank_size: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> None:
        self.batch_size = batch_size
        self.bank_size = max(1, bank_size)
        self.device = device
        self.generator = generator
        self.index_bank = torch.empty(
            (self.bank_size, update_epochs, minibatches, minibatch_size),
            dtype=torch.long,
            device=device,
        )
        self.cursor = 0
        self._fill()

    def _fill(self) -> None:
        for bank_index in range(self.index_bank.shape[0]):
            for epoch in range(self.index_bank.shape[1]):
                epoch_indices = torch.randperm(self.batch_size, device=self.device, generator=self.generator)
                self.index_bank[bank_index, epoch].copy_(
                    epoch_indices.view(self.index_bank.shape[2], self.index_bank.shape[3])
                )

    def next(self) -> torch.Tensor:
        current = self.index_bank[self.cursor]
        self.cursor = (self.cursor + 1) % self.bank_size
        return current


class _StaticGAEPack:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        device: torch.device,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        batch_size: int,
    ) -> None:
        self.device = device
        self.rewards = rewards
        self.dones = dones
        self.values = values
        self.gamma = float(config["gamma"])
        self.gae_lambda = float(config["gae_lambda"])
        self.run_fn = _maybe_compile_gae(config)
        self.graph_enabled = bool(config.get("graph_gae_pack", False)) and device.type == "cuda"
        self.graph_warmup_updates = int(config.get("graph_warmup_updates", 3))
        self.graph_failed = False
        self.graph_error: str | None = None
        self.captured = False
        self.capture_update = 0
        self.graph: torch.cuda.CUDAGraph | None = None
        self.advantages = torch.empty_like(rewards)
        self.returns = torch.empty_like(rewards)
        self.flat_advantages = torch.empty(batch_size, dtype=torch.float32, device=device)
        self.flat_returns = torch.empty(batch_size, dtype=torch.float32, device=device)
        self.next_value = torch.empty(rewards.shape[1], dtype=torch.float32, device=device)

    def _run_once(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.run_fn(
            advantages=self.advantages,
            returns=self.returns,
            rewards=self.rewards,
            dones=self.dones,
            values=self.values,
            next_value=self.next_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        self.flat_advantages.copy_(self.advantages.reshape(-1))
        self.flat_advantages.sub_(self.flat_advantages.mean())
        self.flat_advantages.div_(self.flat_advantages.std().clamp(min=1e-6))
        self.flat_returns.copy_(self.returns.reshape(-1))
        return self.flat_advantages, self.flat_returns

    def maybe_capture(self, update: int) -> None:
        if not self.graph_enabled or self.graph_failed or self.captured or update < self.graph_warmup_updates:
            return
        try:
            self._run_once()
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._run_once()
            self.captured = True
            self.capture_update = update
        except Exception as exc:
            self.graph_failed = True
            self.graph_error = repr(exc)
            self.graph = None

    def run(self, *, next_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.next_value.copy_(next_value)
        if self.captured and self.graph is not None:
            self.graph.replay()
            return self.flat_advantages, self.flat_returns
        return self._run_once()


def _update_episode_stats(
    *,
    dones: torch.Tensor,
    info: dict[str, torch.Tensor],
    episode_count_t: torch.Tensor,
    episode_coverage_sum_t: torch.Tensor,
    episode_return_sum_t: torch.Tensor,
    episode_length_sum_t: torch.Tensor,
    recent_coverage_t: torch.Tensor,
    recent_return_t: torch.Tensor,
    recent_length_t: torch.Tensor,
    recent_decay: float,
    recent_initialized_t: torch.Tensor,
) -> None:
    done_f = dones.to(torch.float32)
    done_count_t = done_f.sum()
    safe_count_t = done_count_t.clamp(min=1.0)
    done_coverages = info["final_coverage"] * done_f
    done_returns = info["episode_return"] * done_f
    done_lengths = info["episode_length"].to(torch.float32) * done_f
    episode_count_t.add_(done_count_t.to(torch.long))
    episode_coverage_sum_t.add_(done_coverages.sum())
    episode_return_sum_t.add_(done_returns.sum())
    episode_length_sum_t.add_(done_lengths.sum())
    batch_mean_coverage_t = done_coverages.sum() / safe_count_t
    batch_mean_return_t = done_returns.sum() / safe_count_t
    batch_mean_length_t = done_lengths.sum() / safe_count_t
    recent_coverage_t.copy_(
        torch.where(
            recent_initialized_t,
            recent_decay * recent_coverage_t + (1.0 - recent_decay) * batch_mean_coverage_t,
            batch_mean_coverage_t,
        )
    )
    recent_return_t.copy_(
        torch.where(
            recent_initialized_t,
            recent_decay * recent_return_t + (1.0 - recent_decay) * batch_mean_return_t,
            batch_mean_return_t,
        )
    )
    recent_length_t.copy_(
        torch.where(
            recent_initialized_t,
            recent_decay * recent_length_t + (1.0 - recent_decay) * batch_mean_length_t,
            batch_mean_length_t,
        )
    )
    recent_initialized_t.logical_or_(done_count_t > 0)


def _device_stats_to_host(stats: dict[str, torch.Tensor]) -> dict[str, float]:
    if not stats:
        return {}
    keys = list(stats)
    packed = torch.stack([stats[key].to(torch.float32) for key in keys]).detach().cpu()
    return {key: float(packed[index].item()) for index, key in enumerate(keys)}


def _ppo_loss(
    *,
    model: SnakePolicy,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    masks: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    autocast_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=amp_dtype):
        logits, new_values = model(obs)
        dist = _masked_dist(logits, masks)
        new_logprobs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        ratios = (new_logprobs - old_logprobs).exp()
        unclipped = ratios * advantages
        clipped = torch.clamp(
            ratios,
            1.0 - float(config["clip_coef"]),
            1.0 + float(config["clip_coef"]),
        ) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()
        new_values = new_values.to(torch.float32)
        value_loss_unclipped = (new_values - returns).pow(2)
        if bool(config.get("use_value_clipping", True)):
            value_delta = new_values - old_values
            clipped_values = old_values + value_delta.clamp(
                -float(config["clip_coef"]),
                float(config["clip_coef"]),
            )
            value_loss_clipped = (clipped_values - returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = 0.5 * value_loss_unclipped.mean()
        loss = (
            policy_loss
            + float(config["value_coef"]) * value_loss
            - float(config["entropy_coef"]) * entropy
        )
    return loss, policy_loss, value_loss, entropy


class _StaticLearner:
    def __init__(
        self,
        *,
        model: SnakePolicy,
        optimizer: torch.optim.Optimizer,
        config: dict[str, Any],
        device: torch.device,
        board_size: int,
        minibatch_size: int,
        amp_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.minibatch_size = minibatch_size
        self.amp_dtype = amp_dtype
        self.autocast_enabled = autocast_enabled
        self.disable_grad_clip = bool(config.get("graph_disable_grad_clip", False))
        self.workspace = _MinibatchWorkspace(
            device=device,
            board_size=board_size,
            minibatch_size=minibatch_size,
        )
        self.policy_loss_tensor = torch.zeros((), dtype=torch.float32, device=device)
        self.value_loss_tensor = torch.zeros((), dtype=torch.float32, device=device)
        self.entropy_tensor = torch.zeros((), dtype=torch.float32, device=device)
        self.graph_enabled = bool(config.get("graph_learner", False)) and device.type == "cuda"
        self.graph_whole_learner = self.graph_enabled and bool(config.get("graph_whole_learner", False))
        self.graph_warmup_updates = int(config.get("graph_warmup_updates", 3))
        self.graph_failed = False
        self.graph_error: str | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.captured = False
        self.capture_update = 0
        self.batch_obs: torch.Tensor | None = None
        self.batch_actions: torch.Tensor | None = None
        self.batch_old_logprobs: torch.Tensor | None = None
        self.batch_advantages: torch.Tensor | None = None
        self.batch_returns: torch.Tensor | None = None
        self.batch_old_values: torch.Tensor | None = None
        self.batch_masks: torch.Tensor | None = None
        self.index_bank: torch.Tensor | None = None
        if self.graph_whole_learner:
            update_epochs = int(config["update_epochs"])
            minibatches = int(config["minibatches"])
            self.index_bank = torch.empty((update_epochs, minibatches, minibatch_size), dtype=torch.long, device=device)

    def load_minibatch(
        self,
        *,
        indices: torch.Tensor,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        masks: torch.Tensor,
    ) -> None:
        self.workspace.load(
            indices=indices,
            obs=obs,
            actions=actions,
            old_logprobs=old_logprobs,
            advantages=advantages,
            returns=returns,
            old_values=old_values,
            masks=masks,
        )

    def set_whole_batch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        masks: torch.Tensor,
    ) -> None:
        self.batch_obs = obs
        self.batch_actions = actions
        self.batch_old_logprobs = old_logprobs
        self.batch_advantages = advantages
        self.batch_returns = returns
        self.batch_old_values = old_values
        self.batch_masks = masks

    def load_index_bank(self, index_bank: torch.Tensor) -> None:
        if self.index_bank is not None:
            self.index_bank.copy_(index_bank)

    def _step_once(self) -> None:
        loss, policy_loss, value_loss, entropy = _ppo_loss(
            model=self.model,
            obs=self.workspace.obs,
            actions=self.workspace.actions,
            old_logprobs=self.workspace.old_logprobs,
            advantages=self.workspace.advantages,
            returns=self.workspace.returns,
            old_values=self.workspace.old_values,
            masks=self.workspace.masks,
            config=self.config,
            device=self.device,
            autocast_enabled=self.autocast_enabled,
            amp_dtype=self.amp_dtype,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not self.disable_grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["max_grad_norm"]))
        self.optimizer.step()
        self.policy_loss_tensor.copy_(policy_loss.detach().to(torch.float32))
        self.value_loss_tensor.copy_(value_loss.detach().to(torch.float32))
        self.entropy_tensor.copy_(entropy.detach().to(torch.float32))

    def _step_whole_learner(self) -> None:
        assert self.index_bank is not None
        assert self.batch_obs is not None
        assert self.batch_actions is not None
        assert self.batch_old_logprobs is not None
        assert self.batch_advantages is not None
        assert self.batch_returns is not None
        assert self.batch_old_values is not None
        assert self.batch_masks is not None
        for epoch in range(self.index_bank.shape[0]):
            for minibatch in range(self.index_bank.shape[1]):
                self.workspace.load(
                    indices=self.index_bank[epoch, minibatch],
                    obs=self.batch_obs,
                    actions=self.batch_actions,
                    old_logprobs=self.batch_old_logprobs,
                    advantages=self.batch_advantages,
                    returns=self.batch_returns,
                    old_values=self.batch_old_values,
                    masks=self.batch_masks,
                )
                self._step_once()

    def maybe_capture(self, update: int) -> None:
        if not self.graph_enabled or self.graph_failed or self.captured or update < self.graph_warmup_updates:
            return
        try:
            if self.graph_whole_learner:
                self._step_whole_learner()
            else:
                self._step_once()
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                if self.graph_whole_learner:
                    self._step_whole_learner()
                else:
                    self._step_once()
            self.captured = True
            self.capture_update = update
        except Exception as exc:
            self.graph_failed = True
            self.graph_error = repr(exc)
            self.graph = None

    def run(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.captured and self.graph is not None:
            self.graph.replay()
        elif self.graph_whole_learner:
            self._step_whole_learner()
        else:
            self._step_once()
        return self.policy_loss_tensor, self.value_loss_tensor, self.entropy_tensor


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
    model = _maybe_compile_model(model, config)

    graph_learner_enabled = bool(config.get("graph_learner", False)) and device.type == "cuda"
    optimizer_kwargs: dict[str, Any] = {"lr": float(config["learning_rate"])}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
        if graph_learner_enabled:
            optimizer_kwargs["capturable"] = True
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
    advantages_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    returns_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    masks_buf = torch.zeros((rollout_steps, num_envs, 4), dtype=torch.bool, device=device)

    best_eval = float("-inf")
    success_reached = False
    metrics_path = run_dir / str(config["metrics_file"])
    checkpoint_dir = run_dir / str(config["checkpoint_dir"])
    scaler_enabled = (
        bool(config.get("amp", False))
        and device.type == "cuda"
        and not (graph_learner_enabled and bool(config.get("graph_disable_grad_scaler", True)))
    )
    amp_dtype_name = str(config.get("amp_dtype", "float16")).lower()
    amp_dtype = torch.float16 if amp_dtype_name == "float16" else torch.bfloat16
    rollout_generator: torch.Generator | None = None
    shuffle_generator: torch.Generator | None = None
    if device.type == "cuda":
        rollout_generator = torch.Generator(device=device)
        rollout_generator.manual_seed(int(config["seed"]) + 20_001)
        shuffle_generator = torch.Generator(device=device)
        shuffle_generator.manual_seed(int(config["seed"]) + 30_001)

    episode_count_t = torch.zeros((), dtype=torch.long, device=device)
    episode_coverage_sum_t = torch.zeros((), dtype=torch.float32, device=device)
    episode_return_sum_t = torch.zeros((), dtype=torch.float32, device=device)
    episode_length_sum_t = torch.zeros((), dtype=torch.float32, device=device)
    recent_coverage_t = torch.zeros((), dtype=torch.float32, device=device)
    recent_return_t = torch.zeros((), dtype=torch.float32, device=device)
    recent_length_t = torch.zeros((), dtype=torch.float32, device=device)
    recent_initialized_t = torch.zeros((), dtype=torch.bool, device=device)
    recent_decay = 0.9
    start_time = time.perf_counter()
    last_console_log = start_time
    flat_obs = obs_buf.view(batch_size, env.board_size, env.board_size)
    flat_actions = actions_buf.view(batch_size)
    flat_logprobs = logprobs_buf.view(batch_size)
    flat_rewards = rewards_buf.view(batch_size)
    flat_dones = dones_buf.view(batch_size)
    flat_values = values_buf.view(batch_size)
    flat_advantages = advantages_buf.view(batch_size)
    flat_returns = returns_buf.view(batch_size)
    flat_masks = masks_buf.view(batch_size, 4)
    policy_loss_tensor = torch.zeros((), dtype=torch.float32, device=device)
    value_loss_tensor = torch.zeros((), dtype=torch.float32, device=device)
    entropy_tensor = torch.zeros((), dtype=torch.float32, device=device)
    static_minibatch_buffers = bool(config.get("static_minibatch_buffers", True))
    metrics_interval_updates = max(0, int(config.get("metrics_interval_updates", 1)))
    gae_pack = _StaticGAEPack(
        config=config,
        device=device,
        rewards=rewards_buf,
        dones=dones_buf,
        values=values_buf,
        batch_size=batch_size,
    )
    permutation_bank = _PermutationBank(
        batch_size=batch_size,
        minibatches=minibatches,
        minibatch_size=minibatch_size,
        update_epochs=int(config["update_epochs"]),
        bank_size=int(config.get("permutation_bank_size", 1)),
        device=device,
        generator=shuffle_generator,
    )
    learner = _StaticLearner(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
        board_size=env.board_size,
        minibatch_size=minibatch_size,
        amp_dtype=amp_dtype,
        autocast_enabled=bool(config.get("amp", False)) and device.type == "cuda",
    )
    if learner.graph_whole_learner:
        learner.set_whole_batch(
            obs=flat_obs,
            actions=flat_actions,
            old_logprobs=flat_logprobs,
            advantages=flat_advantages,
            returns=flat_returns,
            old_values=flat_values,
            masks=flat_masks,
        )

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

            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=scaler_enabled, dtype=amp_dtype):
                    logits, values = model(obs)
                actions, logprobs = _sample_masked_actions(logits, mask, generator=rollout_generator)

            actions_buf[step].copy_(actions)
            logprobs_buf[step].copy_(logprobs)
            values_buf[step].copy_(values.to(torch.float32))

            obs, rewards, dones, info = env.step(actions)
            rewards_buf[step].copy_(rewards)
            dones_buf[step].copy_(dones.to(torch.float32))
            _update_episode_stats(
                dones=dones,
                info=info,
                episode_count_t=episode_count_t,
                episode_coverage_sum_t=episode_coverage_sum_t,
                episode_return_sum_t=episode_return_sum_t,
                episode_length_sum_t=episode_length_sum_t,
                recent_coverage_t=recent_coverage_t,
                recent_return_t=recent_return_t,
                recent_length_t=recent_length_t,
                recent_decay=recent_decay,
                recent_initialized_t=recent_initialized_t,
            )
            env.reset(dones)
            obs = env.observe()
        if profile_update:
            rollout_seconds = _sync_perf_counter(device) - rollout_start

        if profile_update:
            gae_start = _sync_perf_counter(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=scaler_enabled, dtype=amp_dtype):
                _, next_values = model(obs)

        gae_pack.maybe_capture(update)
        flat_advantages, flat_returns = gae_pack.run(next_value=next_values.to(torch.float32))
        if profile_update:
            gae_seconds = _sync_perf_counter(device) - gae_start

        b_obs = flat_obs
        b_actions = flat_actions
        b_logprobs = flat_logprobs
        b_advantages = flat_advantages
        b_returns = flat_returns
        b_values = flat_values
        b_masks = flat_masks

        if profile_update:
            optimize_start = _sync_perf_counter(device)
        index_bank = permutation_bank.next()
        if learner.graph_whole_learner:
            learner.set_whole_batch(
                obs=b_obs,
                actions=b_actions,
                old_logprobs=b_logprobs,
                advantages=b_advantages,
                returns=b_returns,
                old_values=b_values,
                masks=b_masks,
            )
            learner.load_index_bank(index_bank)
            learner.maybe_capture(update)
            policy_loss_tensor, value_loss_tensor, entropy_tensor = learner.run()
        else:
            for epoch in range(int(config["update_epochs"])):
                for minibatch in range(minibatches):
                    mb_indices = index_bank[epoch, minibatch]
                    if static_minibatch_buffers:
                        learner.load_minibatch(
                            indices=mb_indices,
                            obs=b_obs,
                            actions=b_actions,
                            old_logprobs=b_logprobs,
                            advantages=b_advantages,
                            returns=b_returns,
                            old_values=b_values,
                            masks=b_masks,
                        )
                        learner.maybe_capture(update)
                        if learner.captured or learner.graph_failed or graph_learner_enabled:
                            policy_loss_tensor, value_loss_tensor, entropy_tensor = learner.run()
                        else:
                            loss, policy_loss, value_loss, entropy = _ppo_loss(
                                model=model,
                                obs=learner.workspace.obs,
                                actions=learner.workspace.actions,
                                old_logprobs=learner.workspace.old_logprobs,
                                advantages=learner.workspace.advantages,
                                returns=learner.workspace.returns,
                                old_values=learner.workspace.old_values,
                                masks=learner.workspace.masks,
                                config=config,
                                device=device,
                                autocast_enabled=scaler_enabled,
                                amp_dtype=amp_dtype,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
                            optimizer.step()
                            policy_loss_tensor.copy_(policy_loss.detach().to(torch.float32))
                            value_loss_tensor.copy_(value_loss.detach().to(torch.float32))
                            entropy_tensor.copy_(entropy.detach().to(torch.float32))
                    else:
                        mb_obs = b_obs[mb_indices]
                        mb_actions = b_actions[mb_indices]
                        mb_old_logprobs = b_logprobs[mb_indices]
                        mb_advantages = b_advantages[mb_indices]
                        mb_returns = b_returns[mb_indices]
                        mb_old_values = b_values[mb_indices]
                        mb_masks = b_masks[mb_indices]
                        loss, policy_loss, value_loss, entropy = _ppo_loss(
                            model=model,
                            obs=mb_obs,
                            actions=mb_actions,
                            old_logprobs=mb_old_logprobs,
                            advantages=mb_advantages,
                            returns=mb_returns,
                            old_values=mb_old_values,
                            masks=mb_masks,
                            config=config,
                            device=device,
                            autocast_enabled=scaler_enabled,
                            amp_dtype=amp_dtype,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
                        optimizer.step()
                        policy_loss_tensor.copy_(policy_loss.detach().to(torch.float32))
                        value_loss_tensor.copy_(value_loss.detach().to(torch.float32))
                        entropy_tensor.copy_(entropy.detach().to(torch.float32))
        if profile_update:
            optimize_seconds = _sync_perf_counter(device) - optimize_start

        metrics: dict[str, Any] | None = None
        now = time.perf_counter()
        scheduled_eval = _is_eval_scheduled(
            update,
            total_updates=int(config["total_updates"]),
            eval_interval=int(config["eval_interval"]),
            eval_after_update=int(config.get("eval_after_update", 0)),
            eval_interval_after=int(config.get("eval_interval_after", 0)),
        )
        should_emit_console = (
            now - last_console_log >= float(config["console_log_interval_seconds"])
            or scheduled_eval
            or update == int(config["total_updates"])
        )
        should_emit_metrics = (
            metrics_interval_updates > 0 and update % metrics_interval_updates == 0
        ) or scheduled_eval or update == int(config["total_updates"]) or profile_update
        need_host_metrics = should_emit_console or should_emit_metrics

        recent_coverage = 0.0
        if need_host_metrics:
            host_stats = _device_stats_to_host(
                {
                    "policy_loss": policy_loss_tensor,
                    "value_loss": value_loss_tensor,
                    "entropy": entropy_tensor,
                    "episodes_completed": episode_count_t,
                    "mean_recent_final_coverage": recent_coverage_t,
                    "mean_recent_episode_return": recent_return_t,
                    "mean_recent_episode_length": recent_length_t,
                    "mean_completed_final_coverage": episode_coverage_sum_t / episode_count_t.clamp(min=1),
                    "mean_completed_episode_return": episode_return_sum_t / episode_count_t.clamp(min=1),
                    "mean_completed_episode_length": episode_length_sum_t / episode_count_t.clamp(min=1),
                }
            )
            recent_coverage = host_stats["mean_recent_final_coverage"]
            metrics = {
                "update": update,
                "elapsed_seconds": now - start_time,
                "update_seconds": now - update_start,
                "env_steps": update * rollout_steps * num_envs,
                "policy_loss": host_stats["policy_loss"],
                "value_loss": host_stats["value_loss"],
                "entropy": host_stats["entropy"],
                "episodes_completed": int(host_stats["episodes_completed"]),
                "mean_completed_final_coverage": host_stats["mean_completed_final_coverage"],
                "mean_completed_episode_return": host_stats["mean_completed_episode_return"],
                "mean_completed_episode_length": host_stats["mean_completed_episode_length"],
                "mean_recent_final_coverage": host_stats["mean_recent_final_coverage"],
                "mean_recent_episode_return": host_stats["mean_recent_episode_return"],
                "mean_recent_episode_length": host_stats["mean_recent_episode_length"],
            }
            if profile_update:
                metrics["rollout_seconds"] = rollout_seconds
                metrics["gae_seconds"] = gae_seconds
                metrics["optimize_seconds"] = optimize_seconds
                metrics["learner_samples_per_second"] = (
                    (int(config["update_epochs"]) * batch_size) / max(optimize_seconds, 1e-6)
                )
            if learner.captured:
                metrics["graph_capture_update"] = learner.capture_update
            if learner.graph_failed and learner.graph_error is not None:
                metrics["graph_error"] = learner.graph_error
            if gae_pack.captured:
                metrics["gae_graph_capture_update"] = gae_pack.capture_update
            if gae_pack.graph_failed and gae_pack.graph_error is not None:
                metrics["gae_graph_error"] = gae_pack.graph_error

        if scheduled_eval and _should_eval(
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
            if metrics is None:
                metrics = {
                    "update": update,
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "update_seconds": time.perf_counter() - update_start,
                    "env_steps": update * rollout_steps * num_envs,
                }
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
            if metrics is None:
                metrics = {
                    "update": update,
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "update_seconds": time.perf_counter() - update_start,
                    "env_steps": update * rollout_steps * num_envs,
                }
            metrics["checkpoint_seconds"] = checkpoint_seconds

        if metrics is not None:
            append_jsonl(metrics_path, metrics)
        if metrics is not None and (
            should_emit_console
            or "eval_mean_final_coverage" in metrics
            or update == int(config["total_updates"])
        ):
            logger.info(
                "update=%s env_steps=%s recent_cov=%.3f recent_return=%.3f eval_cov=%s sps=%.0f",
                update,
                metrics["env_steps"],
                metrics.get("mean_recent_final_coverage", 0.0),
                metrics.get("mean_recent_episode_return", 0.0),
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
