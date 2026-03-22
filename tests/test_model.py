from __future__ import annotations

import torch

from snake.implementations import implementation4
from snake.model import SnakePolicy


def test_policy_forward_shapes() -> None:
    model = SnakePolicy(board_size=8, trunk_channels=[28, 56], hidden_size=224)
    obs = torch.zeros((3, 8, 8), dtype=torch.uint8)
    logits, values = model(obs)
    assert logits.shape == (3, 4)
    assert values.shape == (3,)


def test_implementation4_build_policy_uses_cnn_model() -> None:
    model = implementation4.build_policy(
        {
            "board_size": 8,
            "trunk_channels": [28, 56],
            "hidden_size": 224,
            "channels_last": True,
        }
    )
    obs = torch.zeros((2, 8, 8), dtype=torch.uint8)
    logits, values = model(obs)
    assert logits.shape == (2, 4)
    assert values.shape == (2,)
