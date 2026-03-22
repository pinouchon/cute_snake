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
    if "env" not in config and "ppo" not in config and "model" not in config:
        return deepcopy(config)

    env = config.get("env", {})
    model = config.get("model", {})
    ppo = config.get("ppo", {})
    runtime = config.get("runtime", {})
    logging = config.get("logging", {})

    return {
        "project": config.get("project", "cute-snake"),
        "seed": int(config.get("seed", 0)),
        "device": config.get("device", "cuda"),
        "run_root": config.get("run_root", "runs"),
        "init_checkpoint": config.get("init_checkpoint"),
        "eval_episodes": int(config.get("eval_episodes", 5)),
        "eval_interval": int(config.get("eval_interval", 10)),
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
        "model_type": str(config.get("model_type", model.get("type", "cnn"))),
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
        "learning_rate": float(config.get("learning_rate", ppo.get("learning_rate", ppo.get("lr", 3e-4)))),
        "max_grad_norm": float(config.get("max_grad_norm", ppo.get("max_grad_norm", 0.5))),
        "total_updates": int(config.get("total_updates", ppo.get("total_updates", 20))),
        "amp": bool(config.get("amp", runtime.get("amp", True))),
        "console_log_interval_seconds": float(config.get("console_log_interval_seconds", runtime.get("console_log_interval_seconds", 15.0))),
        "save_latest": bool(config.get("save_latest", runtime.get("save_latest", True))),
        "metrics_file": logging.get("metrics_file", "metrics.jsonl"),
        "eval_file": logging.get("eval_file", "eval.json"),
        "train_log": logging.get("train_log", "train.log"),
        "checkpoint_dir": logging.get("checkpoint_dir", "checkpoints"),
        "visualization_dir": logging.get("visualization_dir", "visualizations"),
        "profiler_dir": logging.get("profiler_dir", "profiler"),
    }
