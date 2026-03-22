from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def load_policy_state(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    patched_state = {}
    for key, value in state_dict.items():
        normalized_key = key.removeprefix("_orig_mod.")
        patched_state[normalized_key] = value
    model_state = model.state_dict()
    if "board_embedding.weight" in model_state and "board_embedding.weight" not in patched_state:
        patched_state["board_embedding.weight"] = model_state["board_embedding.weight"]
    model.load_state_dict(patched_state)


class SnakePolicy(nn.Module):
    def __init__(
        self,
        board_size: int = 8,
        trunk_channels: list[int] | tuple[int, int] = (32, 64),
        hidden_size: int = 128,
        channels_last: bool = False,
    ) -> None:
        super().__init__()
        if not trunk_channels:
            raise ValueError("trunk_channels must not be empty")

        self.channels_last = channels_last
        self.board_embedding = nn.Embedding(7, 7)
        with torch.no_grad():
            self.board_embedding.weight.copy_(torch.eye(7, dtype=torch.float32))
        self.board_embedding.weight.requires_grad_(False)

        channels = [7, *(int(channel) for channel in trunk_channels)]
        blocks: list[nn.Module] = []
        for in_channels, out_channels in zip(channels, channels[1:], strict=False):
            blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            blocks.append(nn.ReLU())
        self.conv_stack = nn.Sequential(*blocks)
        self.fc = nn.Linear(channels[-1] * board_size * board_size, hidden_size)
        self.actor = nn.Linear(hidden_size, 4)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.board_embedding(obs.long()).permute(0, 3, 1, 2)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x = self.conv_stack(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc(x))
        return self.actor(x), self.critic(x).squeeze(-1)
