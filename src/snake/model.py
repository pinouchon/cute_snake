from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def load_policy_state(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    patched_state = dict(state_dict)
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
        model_type: str = "cnn",
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        channels_last: bool = False,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.model_type = model_type
        self.channels_last = channels_last
        self.board_embedding = nn.Embedding(7, 7)
        with torch.no_grad():
            self.board_embedding.weight.copy_(torch.eye(7, dtype=torch.float32))
        self.board_embedding.weight.requires_grad_(False)
        if model_type == "cnn":
            if len(trunk_channels) < 1:
                raise ValueError(f"Expected at least 1 trunk channel for cnn, got {trunk_channels!r}")
            channels = [7, *(int(channel) for channel in trunk_channels)]
            convs: list[nn.Module] = []
            for in_channels, out_channels in zip(channels, channels[1:], strict=False):
                convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                convs.append(nn.ReLU())
            self.conv_stack = nn.Sequential(*convs)
            last_channels = channels[-1]
            self.fc = nn.Linear(last_channels * board_size * board_size, hidden_size)
        elif model_type == "mlp":
            flat_features = 7 * board_size * board_size
            self.fc1 = nn.Linear(flat_features, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
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
        elif model_type == "axial":
            self.token_projection = nn.Linear(7, hidden_size)
            self.row_embedding = nn.Parameter(torch.zeros(1, board_size, 1, hidden_size))
            self.col_embedding = nn.Parameter(torch.zeros(1, 1, board_size, hidden_size))
            self.axial_layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "row_norm": nn.LayerNorm(hidden_size),
                            "row_attn": nn.MultiheadAttention(
                                embed_dim=hidden_size,
                                num_heads=transformer_heads,
                                dropout=0.0,
                                batch_first=True,
                            ),
                            "col_norm": nn.LayerNorm(hidden_size),
                            "col_attn": nn.MultiheadAttention(
                                embed_dim=hidden_size,
                                num_heads=transformer_heads,
                                dropout=0.0,
                                batch_first=True,
                            ),
                            "mlp_norm": nn.LayerNorm(hidden_size),
                            "mlp": nn.Sequential(
                                nn.Linear(hidden_size, hidden_size * 2),
                                nn.GELU(),
                                nn.Linear(hidden_size * 2, hidden_size),
                            ),
                        }
                    )
                    for _ in range(transformer_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.actor = nn.Linear(hidden_size, 4)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model_type == "cnn":
            x = self.board_embedding(obs.long()).permute(0, 3, 1, 2)
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            x = self.conv_stack(x)
            x = torch.flatten(x, start_dim=1)
            x = F.relu(self.fc(x))
        elif self.model_type == "mlp":
            x = self.board_embedding(obs.long()).reshape(obs.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        elif self.model_type == "axial":
            x = self.board_embedding(obs.long())
            x = self.token_projection(x)
            x = x + self.row_embedding + self.col_embedding
            batch_size = obs.shape[0]
            for layer in self.axial_layers:
                row_input = layer["row_norm"](x).reshape(batch_size * self.board_size, self.board_size, -1)
                if row_input.is_cuda:
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=False,
                        enable_math=True,
                        enable_mem_efficient=False,
                    ):
                        row_output, _ = layer["row_attn"](row_input, row_input, row_input, need_weights=False)
                else:
                    row_output, _ = layer["row_attn"](row_input, row_input, row_input, need_weights=False)
                x = x + row_output.reshape(batch_size, self.board_size, self.board_size, -1)

                col_input = (
                    layer["col_norm"](x)
                    .permute(0, 2, 1, 3)
                    .reshape(batch_size * self.board_size, self.board_size, -1)
                )
                if col_input.is_cuda:
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=False,
                        enable_math=True,
                        enable_mem_efficient=False,
                    ):
                        col_output, _ = layer["col_attn"](col_input, col_input, col_input, need_weights=False)
                else:
                    col_output, _ = layer["col_attn"](col_input, col_input, col_input, need_weights=False)
                col_output = col_output.reshape(batch_size, self.board_size, self.board_size, -1).permute(0, 2, 1, 3)
                x = x + col_output
                x = x + layer["mlp"](layer["mlp_norm"](x))
            x = x.mean(dim=(1, 2))
        else:
            tokens = obs.long().reshape(obs.shape[0], -1)
            x = self.token_embedding(tokens) + self.position_embedding
            x = self.transformer(x)
            x = x.mean(dim=1)
        return self.actor(x), self.critic(x).squeeze(-1)
