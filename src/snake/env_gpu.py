from __future__ import annotations

import random
from typing import Any

import torch


class TorchSnakeBatchEnv:
    def __init__(
        self,
        num_envs: int,
        board_size: int = 8,
        max_steps_since_food: int = 128,
        seed: int = 0,
        device: str | torch.device = "cpu",
        initial_length: int = 3,
        reward_food: float = 1.0,
        reward_death: float = -1.0,
        reward_step: float = -0.01,
    ) -> None:
        self.num_envs = num_envs
        self.board_size = board_size
        self.num_cells = board_size * board_size
        self.max_steps_since_food = max_steps_since_food
        self.seed = seed
        self.device = torch.device(device)
        self.initial_length = initial_length
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.head_codes = torch.tensor([3, 4, 5, 6], dtype=torch.uint8, device=self.device)
        self.offsets = torch.tensor(
            [
                [-1, 0],
                [0, 1],
                [1, 0],
                [0, -1],
            ],
            dtype=torch.long,
            device=self.device,
        )

        self.board = torch.zeros((num_envs, self.num_cells), dtype=torch.uint8, device=self.device)
        self.occupancy = torch.zeros((num_envs, self.num_cells), dtype=torch.bool, device=self.device)
        self.body = torch.zeros((num_envs, self.num_cells), dtype=torch.long, device=self.device)
        self.start = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.length = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.heading = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.food = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.done = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.episode_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.steps_since_food = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.reset_counts = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.rngs = [random.Random(seed + idx * 100_000) for idx in range(num_envs)]

    def reset(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            indices = list(range(self.num_envs))
        else:
            indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        for index in indices:
            episode_index = int(self.reset_counts[index].item())
            self.reset_counts[index] += 1
            self.rngs[index] = random.Random(self.seed + index * 100_000 + episode_index)

            self.board[index].zero_()
            self.occupancy[index].zero_()
            self.body[index].zero_()
            self.start[index] = 0
            self.length[index] = self.initial_length
            self.heading[index] = 1
            self.food[index] = -1
            self.done[index] = False
            self.episode_step[index] = 0
            self.steps_since_food[index] = 0
            self.episode_return[index] = 0.0

            row = self.board_size // 2
            snake = torch.tensor(
                [row * self.board_size + column for column in range(1, 1 + self.initial_length)],
                dtype=torch.long,
                device=self.device,
            )
            self.body[index, : self.initial_length] = snake
            self.occupancy[index, snake] = True
            self.board[index, snake[:-1]] = 2
            self.board[index, snake[-1]] = self.head_codes[1]
            self.food[index] = self._spawn_food(index)
            self.board[index, self.food[index]] = 1
        return self.observe()

    def observe(self) -> torch.Tensor:
        return self.board.view(self.num_envs, self.board_size, self.board_size)

    def action_mask(self) -> torch.Tensor:
        mask = torch.ones((self.num_envs, 4), dtype=torch.bool, device=self.device)
        reverse = (self.heading + 2) % 4
        mask.scatter_(1, reverse.unsqueeze(1), False)
        return mask

    def snapshot(self, index: int = 0) -> dict[str, Any]:
        start = int(self.start[index].item())
        length = int(self.length[index].item())
        snake = [
            int(self.body[index, (start + offset) % self.num_cells].item())
            for offset in range(length)
        ]
        return {
            "snake": snake,
            "heading": int(self.heading[index].item()),
            "food": int(self.food[index].item()),
            "done": bool(self.done[index].item()),
            "episode_step": int(self.episode_step[index].item()),
            "steps_since_food": int(self.steps_since_food[index].item()),
            "episode_return": float(self.episode_return[index].item()),
        }

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        actions = actions.to(self.device, dtype=torch.long)
        active = ~self.done
        reverse = (self.heading + 2) % 4
        actions = torch.where(actions == reverse, self.heading, actions)
        actions = torch.where(active, actions, self.heading)

        old_start = self.start.clone()
        old_length = self.length.clone()
        head_slot = (old_start + old_length - 1) % self.num_cells
        tail_slot = old_start
        env_indices = torch.arange(self.num_envs, device=self.device)
        head_pos = self.body[env_indices, head_slot]
        tail_pos = self.body[env_indices, tail_slot]

        head_row = torch.div(head_pos, self.board_size, rounding_mode="floor")
        head_col = head_pos % self.board_size
        next_row = head_row + self.offsets[actions, 0]
        next_col = head_col + self.offsets[actions, 1]
        wall = (
            (next_row < 0)
            | (next_row >= self.board_size)
            | (next_col < 0)
            | (next_col >= self.board_size)
        ) & active
        safe_next_row = next_row.clamp(0, self.board_size - 1)
        safe_next_col = next_col.clamp(0, self.board_size - 1)
        next_pos = safe_next_row * self.board_size + safe_next_col

        growing = (next_pos == self.food) & active & ~wall
        occupied_next = self.occupancy[env_indices, next_pos] & active & ~wall
        legal_tail = (next_pos == tail_pos) & ~growing
        self_hit = occupied_next & ~legal_tail
        dead = wall | self_hit
        moving = active & ~dead
        non_growing = moving & ~growing

        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        rewards[active] += self.reward_step
        rewards[dead] += self.reward_death

        self.episode_step[active] += 1
        self.heading[moving] = actions[moving]

        if torch.any(non_growing):
            ng_indices = torch.nonzero(non_growing, as_tuple=False).flatten()
            ng_tail = tail_pos[ng_indices]
            self.occupancy[ng_indices, ng_tail] = False
            self.board[ng_indices, ng_tail] = 0
            self.start[ng_indices] = (self.start[ng_indices] + 1) % self.num_cells
            self.steps_since_food[ng_indices] += 1

        if torch.any(moving):
            mv_indices = torch.nonzero(moving, as_tuple=False).flatten()
            old_heads = head_pos[mv_indices]
            self.board[mv_indices, old_heads] = 2

            new_slot = (old_start[mv_indices] + old_length[mv_indices]) % self.num_cells
            new_head = next_pos[mv_indices]
            self.body[mv_indices, new_slot] = new_head
            self.occupancy[mv_indices, new_head] = True
            self.board[mv_indices, new_head] = self.head_codes[self.heading[mv_indices]]

        if torch.any(growing):
            grow_indices = torch.nonzero(growing, as_tuple=False).flatten()
            self.length[grow_indices] += 1
            self.steps_since_food[grow_indices] = 0
            rewards[grow_indices] += self.reward_food
            for index in grow_indices.tolist():
                if int(self.length[index].item()) == self.num_cells:
                    self.food[index] = -1
                    continue
                old_food = int(next_pos[index].item())
                if old_food >= 0:
                    self.board[index, old_food] = self.head_codes[int(self.heading[index].item())]
                new_food = self._spawn_food(index)
                self.food[index] = new_food
                self.board[index, new_food] = 1

        won = moving & (self.length == self.num_cells)
        truncated = moving & ~growing & (self.steps_since_food >= self.max_steps_since_food)
        done = dead | won | truncated
        self.done |= done
        self.episode_return += rewards

        info = {
            "won": won,
            "truncated": truncated,
            "final_coverage": self.length.to(torch.float32) / float(self.num_cells),
            "episode_length": self.episode_step.clone(),
            "episode_return": self.episode_return.clone(),
        }
        return self.observe(), rewards, done, info

    def _spawn_food(self, index: int) -> int:
        occupied = self.occupancy[index].detach().cpu().tolist()
        empty = [cell for cell, used in enumerate(occupied) if not used]
        if not empty:
            return -1
        return self.rngs[index].choice(empty)
