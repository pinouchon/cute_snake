from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from snake.implementations import get_implementation_module


def implementation_name(config: dict[str, Any]) -> str:
    return str(config.get("implementation", "implementation1"))


def build_policy(config: dict[str, Any]) -> torch.nn.Module:
    module = get_implementation_module(config)
    return module.build_policy(config)


def load_policy_from_checkpoint(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    module = get_implementation_module(config)
    model = module.build_policy(config)
    module.load_checkpoint_into_policy(model, checkpoint["model"])
    model.to(device)
    return model


def train(config: dict[str, Any], run_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    module = get_implementation_module(config)
    return module.train(config, run_dir, logger)
