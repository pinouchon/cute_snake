from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping config at {path}, got {type(data)!r}")
    return data


def save_yaml_config(config: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got {override!r}")
        key_path, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor: dict[str, Any] = updated
        parts = key_path.split(".")
        for part in parts[:-1]:
            node = cursor.get(part)
            if node is None:
                node = {}
                cursor[part] = node
            if not isinstance(node, dict):
                raise TypeError(f"Cannot override nested key {key_path!r}")
            cursor = node
        cursor[parts[-1]] = value
    return updated


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    if "env" not in config and "ppo" not in config and "model" not in config and "dqn" not in config:
        return deepcopy(config)

    env = config.get("env", {})
    model = config.get("model", {})
    ppo = config.get("ppo", {})
    dqn = config.get("dqn", {})
    runtime = config.get("runtime", {})
    logging = config.get("logging", {})

    return {
        "project": config.get("project", "cute-snake"),
        "implementation": str(config.get("implementation", "implementation1")),
        "seed": int(config.get("seed", 0)),
        "device": config.get("device", "cuda"),
        "run_root": config.get("run_root", "runs"),
        "init_checkpoint": config.get("init_checkpoint"),
        "eval_episodes": int(config.get("eval_episodes", 5)),
        "eval_interval": int(config.get("eval_interval", 10)),
        "eval_after_update": int(config.get("eval_after_update", 0)),
        "eval_interval_after": int(config.get("eval_interval_after", 0)),
        "eval_recent_coverage_gate": float(config.get("eval_recent_coverage_gate", 0.0)),
        "checkpoint_interval": int(config.get("checkpoint_interval", 10)),
        "success_target": float(config.get("success_target", 0.80)),
        "stop_on_success": bool(config.get("stop_on_success", True)),
        "board_size": int(config.get("board_size", env.get("board_size", 8))),
        "initial_length": int(config.get("initial_length", env.get("initial_length", 3))),
        "food_count": int(config.get("food_count", env.get("food_count", 1))),
        "max_steps_since_food": int(config.get("max_steps_since_food", env.get("starvation_steps", 128))),
        "reward_food": float(config.get("reward_food", env.get("reward_food", 1.0))),
        "reward_death": float(config.get("reward_death", env.get("reward_death", -1.0))),
        "reward_step": float(config.get("reward_step", env.get("reward_step", -0.01))),
        "use_cute_step_core": bool(config.get("use_cute_step_core", env.get("use_cute_step_core", False))),
        "compile_model": bool(config.get("compile_model", model.get("compile_model", runtime.get("compile_model", False)))),
        "matmul_precision": config.get("matmul_precision", runtime.get("matmul_precision", "high")),
        "allow_tf32": bool(config.get("allow_tf32", runtime.get("allow_tf32", True))),
        "cudnn_benchmark": bool(config.get("cudnn_benchmark", runtime.get("cudnn_benchmark", True))),
        "channels_last": bool(config.get("channels_last", model.get("channels_last", runtime.get("channels_last", False)))),
        "amp_dtype": str(config.get("amp_dtype", runtime.get("amp_dtype", "float16"))),
        "model_type": str(config.get("model_type", model.get("type", "cnn"))),
        "trunk_channels": list(config.get("trunk_channels", model.get("trunk_channels", [32, 64]))),
        "hidden_size": int(config.get("hidden_size", model.get("hidden_dim", 128))),
        "transformer_layers": int(config.get("transformer_layers", model.get("transformer_layers", 4))),
        "transformer_heads": int(config.get("transformer_heads", model.get("transformer_heads", 8))),
        "num_envs": int(config.get("num_envs", ppo.get("num_envs", 1024))),
        "rollout_steps": int(config.get("rollout_steps", ppo.get("rollout_steps", ppo.get("rollout_len", 32)))),
        "minibatches": int(config.get("minibatches", ppo.get("minibatches", 4))),
        "update_epochs": int(config.get("update_epochs", ppo.get("update_epochs", 2))),
        "gamma": float(config.get("gamma", ppo.get("gamma", 0.99))),
        "gae_lambda": float(config.get("gae_lambda", ppo.get("gae_lambda", 0.95))),
        "clip_coef": float(config.get("clip_coef", ppo.get("clip_coef", 0.2))),
        "value_coef": float(config.get("value_coef", ppo.get("value_coef", 0.5))),
        "entropy_coef": float(config.get("entropy_coef", ppo.get("entropy_coef", 0.01))),
        "use_value_clipping": bool(config.get("use_value_clipping", ppo.get("use_value_clipping", True))),
        "learning_rate": float(config.get("learning_rate", ppo.get("learning_rate", ppo.get("lr", 3e-4)))),
        "max_grad_norm": float(config.get("max_grad_norm", ppo.get("max_grad_norm", 0.5))),
        "total_updates": int(config.get("total_updates", ppo.get("total_updates", 20))),
        "batch_size": int(config.get("batch_size", dqn.get("batch_size", 1024))),
        "replay_capacity": int(config.get("replay_capacity", dqn.get("replay_capacity", 500_000))),
        "learning_starts": int(config.get("learning_starts", dqn.get("learning_starts", 50_000))),
        "train_steps_per_update": int(config.get("train_steps_per_update", dqn.get("train_steps_per_update", 4))),
        "target_update_interval": int(config.get("target_update_interval", dqn.get("target_update_interval", 2))),
        "target_tau": float(config.get("target_tau", dqn.get("target_tau", 1.0))),
        "epsilon_start": float(config.get("epsilon_start", dqn.get("epsilon_start", 1.0))),
        "epsilon_end": float(config.get("epsilon_end", dqn.get("epsilon_end", 0.05))),
        "epsilon_decay_fraction": float(config.get("epsilon_decay_fraction", dqn.get("epsilon_decay_fraction", 0.2))),
        "double_dqn": bool(config.get("double_dqn", dqn.get("double_dqn", True))),
        "n_step": int(config.get("n_step", dqn.get("n_step", 1))),
        "amp": bool(config.get("amp", runtime.get("amp", True))),
        "console_log_interval_seconds": float(config.get("console_log_interval_seconds", runtime.get("console_log_interval_seconds", 15.0))),
        "profile_interval_updates": int(config.get("profile_interval_updates", runtime.get("profile_interval_updates", 10))),
        "save_latest": bool(config.get("save_latest", runtime.get("save_latest", True))),
        "metrics_file": logging.get("metrics_file", "metrics.jsonl"),
        "eval_file": logging.get("eval_file", "eval.json"),
        "train_log": logging.get("train_log", "train.log"),
        "checkpoint_dir": logging.get("checkpoint_dir", "checkpoints"),
        "visualization_dir": logging.get("visualization_dir", "visualizations"),
        "profiler_dir": logging.get("profiler_dir", "profiler"),
    }
