from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SnakePolicy(nn.Module):
    def __init__(
        self,
        board_size: int = 8,
        hidden_size: int = 128,
        model_type: str = "cnn",
        transformer_layers: int = 4,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.model_type = model_type
        if model_type == "cnn":
            self.conv1 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc = nn.Linear(64 * board_size * board_size, hidden_size)
        elif model_type == "transformer":
            self.token_embedding = nn.Embedding(7, hidden_size)
            self.position_embedding = nn.Parameter(
                torch.zeros(1, board_size * board_size, hidden_size)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=transformer_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
                dropout=0.0,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers,
            )
            self.fc = nn.Linear(hidden_size, hidden_size)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.actor = nn.Linear(hidden_size, 4)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model_type == "cnn":
            x = F.one_hot(obs.long(), num_classes=7).permute(0, 3, 1, 2).to(torch.float32)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = torch.flatten(x, start_dim=1)
        else:
            tokens = obs.long().reshape(obs.shape[0], -1)
            x = self.token_embedding(tokens) + self.position_embedding
            x = self.transformer(x)
            x = x.mean(dim=1)
        x = F.relu(self.fc(x))
        return self.actor(x), self.critic(x).squeeze(-1)
