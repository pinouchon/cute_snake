from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from snake.model import SnakePolicy, load_policy_state
from snake.ppo import train_ppo


def build_policy(config: dict[str, Any]) -> SnakePolicy:
    return SnakePolicy(
        board_size=int(config["board_size"]),
        trunk_channels=list(config.get("trunk_channels", [24, 48])),
        hidden_size=int(config["hidden_size"]),
        model_type=str(config.get("model_type", "axial")),
        transformer_layers=int(config.get("transformer_layers", 2)),
        transformer_heads=int(config.get("transformer_heads", 4)),
    )


def load_checkpoint_into_policy(model: SnakePolicy, state_dict: dict[str, Any]) -> None:
    load_policy_state(model, state_dict)


def train(config: dict[str, Any], run_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    return train_ppo(config, run_dir, logger)
