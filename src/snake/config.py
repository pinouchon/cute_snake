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
    if "env" not in config and "model" not in config and "ppo" not in config and "runtime" not in config:
        normalized = deepcopy(config)
        normalized["implementation"] = "implementation4"
        return normalized

    env = config.get("env", {})
    model = config.get("model", {})
    ppo = config.get("ppo", {})
    runtime = config.get("runtime", {})
    logging = config.get("logging", {})

    return {
        "project": config.get("project", "cute-snake"),
        "implementation": "implementation4",
        "seed": int(config.get("seed", 94)),
        "device": str(config.get("device", "cuda")),
        "run_root": str(config.get("run_root", "runs")),
        "init_checkpoint": config.get("init_checkpoint"),
        "eval_episodes": int(config.get("eval_episodes", 5)),
        "eval_interval": int(config.get("eval_interval", 10)),
        "eval_after_update": int(config.get("eval_after_update", 90)),
        "eval_interval_after": int(config.get("eval_interval_after", 1)),
        "eval_recent_coverage_gate": float(config.get("eval_recent_coverage_gate", 0.65)),
        "checkpoint_interval": int(config.get("checkpoint_interval", 1000)),
        "success_target": float(config.get("success_target", 0.80)),
        "stop_on_success": bool(config.get("stop_on_success", True)),
        "board_size": int(config.get("board_size", env.get("board_size", 8))),
        "initial_length": int(config.get("initial_length", env.get("initial_length", 3))),
        "max_steps_since_food": int(config.get("max_steps_since_food", env.get("starvation_steps", 128))),
        "reward_food": float(config.get("reward_food", env.get("reward_food", 1.0))),
        "reward_death": float(config.get("reward_death", env.get("reward_death", -1.0))),
        "reward_step": float(config.get("reward_step", env.get("reward_step", -0.01))),
        "compile_model": bool(config.get("compile_model", model.get("compile_model", runtime.get("compile_model", False)))),
        "compile_mode": str(config.get("compile_mode", model.get("compile_mode", runtime.get("compile_mode", "reduce-overhead")))),
        "compile_gae": bool(config.get("compile_gae", runtime.get("compile_gae", False))),
        "compile_disable_cudagraphs": bool(
            config.get(
                "compile_disable_cudagraphs",
                model.get("compile_disable_cudagraphs", runtime.get("compile_disable_cudagraphs", False)),
            )
        ),
        "graph_learner": bool(config.get("graph_learner", runtime.get("graph_learner", False))),
        "graph_warmup_updates": int(config.get("graph_warmup_updates", runtime.get("graph_warmup_updates", 2))),
        "graph_disable_grad_clip": bool(
            config.get("graph_disable_grad_clip", runtime.get("graph_disable_grad_clip", False))
        ),
        "startup_prewarm": bool(config.get("startup_prewarm", runtime.get("startup_prewarm", True))),
        "startup_limit_seconds": float(config.get("startup_limit_seconds", runtime.get("startup_limit_seconds", 20.0))),
        "matmul_precision": str(config.get("matmul_precision", runtime.get("matmul_precision", "high"))),
        "allow_tf32": bool(config.get("allow_tf32", runtime.get("allow_tf32", False))),
        "cudnn_benchmark": bool(config.get("cudnn_benchmark", runtime.get("cudnn_benchmark", False))),
        "channels_last": bool(config.get("channels_last", model.get("channels_last", False))),
        "amp": bool(config.get("amp", runtime.get("amp", True))),
        "amp_dtype": str(config.get("amp_dtype", runtime.get("amp_dtype", "bfloat16"))),
        "trunk_channels": list(config.get("trunk_channels", model.get("trunk_channels", [28, 56]))),
        "hidden_size": int(config.get("hidden_size", model.get("hidden_dim", 224))),
        "num_envs": int(config.get("num_envs", ppo.get("num_envs", 4096))),
        "rollout_steps": int(config.get("rollout_steps", ppo.get("rollout_len", 16))),
        "minibatches": int(config.get("minibatches", ppo.get("minibatches", 8))),
        "update_epochs": int(config.get("update_epochs", ppo.get("update_epochs", 2))),
        "gamma": float(config.get("gamma", ppo.get("gamma", 0.995))),
        "gae_lambda": float(config.get("gae_lambda", ppo.get("gae_lambda", 0.98))),
        "clip_coef": float(config.get("clip_coef", ppo.get("clip_coef", 0.2))),
        "value_coef": float(config.get("value_coef", ppo.get("value_coef", 0.5))),
        "entropy_coef": float(config.get("entropy_coef", ppo.get("entropy_coef", 0.005))),
        "use_value_clipping": bool(config.get("use_value_clipping", ppo.get("use_value_clipping", False))),
        "learning_rate": float(config.get("learning_rate", ppo.get("lr", 2.8e-3))),
        "max_grad_norm": float(config.get("max_grad_norm", ppo.get("max_grad_norm", 0.35))),
        "total_updates": int(config.get("total_updates", ppo.get("total_updates", 160))),
        "console_log_interval_seconds": float(
            config.get("console_log_interval_seconds", runtime.get("console_log_interval_seconds", 10.0))
        ),
        "metrics_interval_updates": int(config.get("metrics_interval_updates", runtime.get("metrics_interval_updates", 10))),
        "profile_interval_updates": int(config.get("profile_interval_updates", runtime.get("profile_interval_updates", 0))),
        "save_latest": bool(config.get("save_latest", runtime.get("save_latest", False))),
        "metrics_file": str(logging.get("metrics_file", "metrics.jsonl")),
        "eval_file": str(logging.get("eval_file", "eval.json")),
        "train_log": str(logging.get("train_log", "train.log")),
        "checkpoint_dir": str(logging.get("checkpoint_dir", "checkpoints")),
        "visualization_dir": str(logging.get("visualization_dir", "visualizations")),
        "profiler_dir": str(logging.get("profiler_dir", "profiler")),
    }
