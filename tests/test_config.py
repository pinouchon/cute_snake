from __future__ import annotations

import torch

from snake.config import normalize_config
from snake.implementations.implementation4 import _dg_gate


def test_normalize_config_includes_dg_defaults() -> None:
    config = normalize_config(
        {
            "ppo": {},
            "model": {},
            "env": {},
            "runtime": {},
        }
    )
    assert config["dg_enabled"] is False
    assert config["dg_eta"] == 1.0
    assert config["dg_use_raw_advantage_for_gate"] is True
    assert config["dg_detach_gate"] is True
    assert config["dg_gate_floor"] == 0.0
    assert config["dg_log_metrics"] is True


def test_dg_gate_uses_raw_advantage_when_enabled() -> None:
    new_logprobs = torch.tensor([-2.0, -0.5], dtype=torch.float32)
    raw_advantages = torch.tensor([2.0, -2.0], dtype=torch.float32)
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)
    gated_advantages, stats = _dg_gate(
        new_logprobs=new_logprobs,
        raw_advantages=raw_advantages,
        advantages=advantages,
        config={
            "dg_eta": 1.0,
            "dg_use_raw_advantage_for_gate": True,
            "dg_detach_gate": True,
            "dg_gate_floor": 0.0,
        },
    )
    assert gated_advantages.shape == advantages.shape
    assert gated_advantages[0] > 0.9
    assert gated_advantages[1] > -0.3
    assert "dg_gate_mean" in stats


def test_dg_gate_floor_clamps_gate() -> None:
    gated_advantages, stats = _dg_gate(
        new_logprobs=torch.tensor([-2.0], dtype=torch.float32),
        raw_advantages=torch.tensor([-10.0], dtype=torch.float32),
        advantages=torch.tensor([-1.0], dtype=torch.float32),
        config={
            "dg_eta": 1.0,
            "dg_use_raw_advantage_for_gate": True,
            "dg_detach_gate": True,
            "dg_gate_floor": 0.25,
        },
    )
    assert gated_advantages.item() == -0.25
    assert stats["dg_gate_mean"].item() == 0.25
