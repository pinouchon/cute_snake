from __future__ import annotations

import torch

from snake.implementations import implementation4
from snake.model import SnakePolicy


def test_axial_policy_forward_shapes() -> None:
    model = SnakePolicy(
        board_size=8,
        hidden_size=96,
        model_type="axial",
        transformer_layers=2,
        transformer_heads=4,
    )
    obs = torch.zeros((3, 8, 8), dtype=torch.uint8)
    logits, values = model(obs)
    assert logits.shape == (3, 4)
    assert values.shape == (3,)


def test_implementation4_policy_forward_shapes() -> None:
    model = implementation4.build_policy(
        {
            "board_size": 8,
            "trunk_channels": [24, 48],
            "hidden_size": 192,
            "model_type": "cnn",
            "transformer_layers": 2,
            "transformer_heads": 4,
        }
    )
    obs = torch.zeros((3, 8, 8), dtype=torch.uint8)
    logits, values = model(obs)
    assert logits.shape == (3, 4)
    assert values.shape == (3,)
