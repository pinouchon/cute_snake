from __future__ import annotations

import random
from typing import Any

import torch

from snake.cute_fused_experiment import fused_step_update_cuda
from snake.cute_kernels import HAVE_CUTE, step_core_cuda


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
        use_cute_step_core: bool = False,
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
        self.use_fast_cuda = self.device.type == "cuda"
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
        self.env_indices = torch.arange(num_envs, dtype=torch.long, device=self.device)
        self.initial_row = self.board_size // 2
        self.initial_snake = torch.tensor(
            [self.initial_row * self.board_size + column for column in range(1, 1 + self.initial_length)],
            dtype=torch.long,
            device=self.device,
        )

        self.board = torch.zeros((num_envs, self.num_cells), dtype=torch.uint8, device=self.device)
        self.occupancy = torch.zeros((num_envs, self.num_cells), dtype=torch.bool, device=self.device)
        self.occupancy_i64: torch.Tensor | None = None
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
        self.step_old_start = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.step_old_length = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.step_rewards = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.info_final_coverage = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.rngs = [random.Random(seed + idx * 100_000) for idx in range(num_envs)]
        self.cuda_generator: torch.Generator | None = None
        self.cute_step_enabled = bool(use_cute_step_core) and self.use_fast_cuda and HAVE_CUTE
        self.cute_done: torch.Tensor | None = None
        self.cute_actions: torch.Tensor | None = None
        self.cute_next_pos: torch.Tensor | None = None
        self.cute_wall: torch.Tensor | None = None
        self.cute_growing: torch.Tensor | None = None
        self.cute_dead: torch.Tensor | None = None
        self.cute_moving: torch.Tensor | None = None
        self.cute_new_heading: torch.Tensor | None = None
        self.cute_new_start: torch.Tensor | None = None
        self.cute_new_length: torch.Tensor | None = None
        self.cute_new_episode_step: torch.Tensor | None = None
        self.cute_new_steps_since_food: torch.Tensor | None = None
        self.cute_new_done: torch.Tensor | None = None
        self.cute_new_episode_return: torch.Tensor | None = None
        self.cute_reward: torch.Tensor | None = None
        self.cute_won: torch.Tensor | None = None
        self.cute_truncated: torch.Tensor | None = None
        if self.use_fast_cuda:
            self.cuda_generator = torch.Generator(device=self.device)
            self.cuda_generator.manual_seed(seed)
        if self.cute_step_enabled:
            self.occupancy_i64 = torch.zeros((num_envs, self.num_cells), dtype=torch.int64, device=self.device)
            self.cute_done = torch.empty(num_envs, dtype=torch.int64, device=self.device)
            self.cute_actions = torch.empty(num_envs, dtype=torch.long, device=self.device)
            self.cute_next_pos = torch.empty(num_envs, dtype=torch.long, device=self.device)
            self.cute_wall = torch.empty(num_envs, dtype=torch.int64, device=self.device)
            self.cute_growing = torch.empty(num_envs, dtype=torch.int64, device=self.device)
            self.cute_dead = torch.empty(num_envs, dtype=torch.int64, device=self.device)
            self.cute_moving = torch.empty(num_envs, dtype=torch.int64, device=self.device)
            self.cute_new_heading = torch.empty(num_envs, dtype=torch.int64, device=self.device)
            self.cute_new_start = torch.empty(num_envs, dtype=torch.int64, device=self.device)
            self.cute_new_length = torch.empty(num_envs, dtype=torch.int64, device=self.device)
            self.cute_new_episode_step = torch.empty(num_envs, dtype=torch.int64, device=self.device)
            self.cute_new_steps_since_food = torch.empty(num_envs, dtype=torch.int64, device=self.device)
            self.cute_new_done = torch.empty(num_envs, dtype=torch.int64, device=self.device)
            self.cute_new_episode_return = torch.empty(num_envs, dtype=torch.float32, device=self.device)
            self.cute_reward = torch.empty(num_envs, dtype=torch.float32, device=self.device)
            self.cute_won = torch.empty(num_envs, dtype=torch.int64, device=self.device)
            self.cute_truncated = torch.empty(num_envs, dtype=torch.int64, device=self.device)

    def reset(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_fast_cuda:
            return self._reset_cuda(mask)

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
            if self.occupancy_i64 is not None:
                self.occupancy_i64[index].zero_()
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
            if self.occupancy_i64 is not None:
                self.occupancy_i64[index, snake] = 1
            self.board[index, snake[:-1]] = 2
            self.board[index, snake[-1]] = self.head_codes[1]
            self.food[index] = self._spawn_food(index)
            self.board[index, self.food[index]] = 1
        return self.observe()

    def _reset_cuda(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            indices = self.env_indices
        else:
            indices = torch.nonzero(mask.to(self.device), as_tuple=False).flatten()
        if indices.numel() == 0:
            return self.observe()

        self.reset_counts[indices] += 1
        self.board[indices] = 0
        self.occupancy[indices] = False
        if self.occupancy_i64 is not None:
            self.occupancy_i64[indices] = 0
        self.body[indices] = 0
        self.start[indices] = 0
        self.length[indices] = self.initial_length
        self.heading[indices] = 1
        self.food[indices] = -1
        self.done[indices] = False
        self.episode_step[indices] = 0
        self.steps_since_food[indices] = 0
        self.episode_return[indices] = 0.0

        initial_cells = self.initial_snake.unsqueeze(0).expand(indices.numel(), -1)
        self.body[indices, : self.initial_length] = initial_cells
        self.occupancy[indices.unsqueeze(1), initial_cells] = True
        if self.occupancy_i64 is not None:
            self.occupancy_i64[indices.unsqueeze(1), initial_cells] = 1
        if self.initial_length > 1:
            self.board[indices.unsqueeze(1), initial_cells[:, :-1]] = 2
        self.board[indices, initial_cells[:, -1]] = self.head_codes[1]

        food = self._spawn_food_many(indices)
        valid = food >= 0
        self.food[indices] = food
        if valid.any():
            valid_indices = indices[valid]
            self.board[valid_indices, food[valid]] = 1
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

        self.step_old_start.copy_(self.start)
        self.step_old_length.copy_(self.length)
        old_start = self.step_old_start
        old_length = self.step_old_length
        head_slot = (old_start + old_length - 1) % self.num_cells
        tail_slot = old_start
        head_pos = self.body[self.env_indices, head_slot]
        tail_pos = self.body[self.env_indices, tail_slot]

        won: torch.Tensor
        truncated: torch.Tensor
        if self.cute_step_enabled:
            try:
                assert self.cute_done is not None
                assert self.cute_actions is not None
                assert self.cute_next_pos is not None
                assert self.cute_wall is not None
                assert self.cute_growing is not None
                assert self.cute_dead is not None
                assert self.cute_moving is not None
                assert self.cute_new_heading is not None
                assert self.cute_new_start is not None
                assert self.cute_new_length is not None
                assert self.cute_new_episode_step is not None
                assert self.cute_new_steps_since_food is not None
                assert self.cute_new_done is not None
                assert self.cute_new_episode_return is not None
                assert self.cute_reward is not None
                assert self.cute_won is not None
                assert self.cute_truncated is not None
                assert self.occupancy_i64 is not None
                self.cute_done.copy_(self.done.to(torch.int64))
                fused_step_update_cuda(
                    actions=actions,
                    heading=self.heading,
                    head_pos=head_pos,
                    tail_pos=tail_pos,
                    food=self.food,
                    done=self.cute_done,
                    occupancy=self.occupancy_i64,
                    start=self.start,
                    length=self.length,
                    episode_step=self.episode_step,
                    steps_since_food=self.steps_since_food,
                    episode_return=self.episode_return,
                    board_size=self.board_size,
                    max_steps_since_food=self.max_steps_since_food,
                    reward_food=self.reward_food,
                    reward_death=self.reward_death,
                    reward_step=self.reward_step,
                    sanitized_actions=self.cute_actions,
                    next_pos=self.cute_next_pos,
                    wall=self.cute_wall,
                    growing=self.cute_growing,
                    dead=self.cute_dead,
                    moving=self.cute_moving,
                    new_heading=self.cute_new_heading,
                    new_start=self.cute_new_start,
                    new_length=self.cute_new_length,
                    new_episode_step=self.cute_new_episode_step,
                    new_steps_since_food=self.cute_new_steps_since_food,
                    new_done=self.cute_new_done,
                    new_episode_return=self.cute_new_episode_return,
                    reward=self.cute_reward,
                    won=self.cute_won,
                    truncated=self.cute_truncated,
                )
                actions = self.cute_actions
                next_pos = self.cute_next_pos
                wall = self.cute_wall.to(torch.bool)
                growing = self.cute_growing.to(torch.bool)
                dead = self.cute_dead.to(torch.bool)
                moving = self.cute_moving.to(torch.bool)
                rewards = self.cute_reward
                won = self.cute_won.to(torch.bool)
                truncated = self.cute_truncated.to(torch.bool)
                self.heading.copy_(self.cute_new_heading)
                self.start.copy_(self.cute_new_start)
                self.length.copy_(self.cute_new_length)
                self.episode_step.copy_(self.cute_new_episode_step)
                self.steps_since_food.copy_(self.cute_new_steps_since_food)
                self.done.copy_(self.cute_new_done.to(torch.bool))
                self.episode_return.copy_(self.cute_new_episode_return)
            except Exception:
                self.cute_step_enabled = False
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
                occupied_next = self.occupancy[self.env_indices, next_pos] & active & ~wall
                legal_tail = (next_pos == tail_pos) & ~growing
                self_hit = occupied_next & ~legal_tail
                dead = wall | self_hit
                moving = active & ~dead
                rewards = self.step_rewards
                rewards.zero_()
                rewards[active] += self.reward_step
                rewards[dead] += self.reward_death
                self.episode_step[active] += 1
                self.heading[moving] = actions[moving]
                won = moving & (self.length == self.num_cells)
                truncated = moving & ~growing & (self.steps_since_food >= self.max_steps_since_food)
        else:
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
            occupied_next = self.occupancy[self.env_indices, next_pos] & active & ~wall
            legal_tail = (next_pos == tail_pos) & ~growing
            self_hit = occupied_next & ~legal_tail
            dead = wall | self_hit
            moving = active & ~dead
            rewards = self.step_rewards
            rewards.zero_()
            rewards[active] += self.reward_step
            rewards[dead] += self.reward_death
            self.episode_step[active] += 1
            self.heading[moving] = actions[moving]
            won = moving & (self.length == self.num_cells)
            truncated = moving & ~growing & (self.steps_since_food >= self.max_steps_since_food)
        non_growing = moving & ~growing

        ng_indices = torch.nonzero(non_growing, as_tuple=False).flatten()
        if ng_indices.numel() > 0:
            ng_tail = tail_pos[ng_indices]
            self.occupancy[ng_indices, ng_tail] = False
            if self.occupancy_i64 is not None:
                self.occupancy_i64[ng_indices, ng_tail] = 0
            self.board[ng_indices, ng_tail] = 0
            if not self.cute_step_enabled:
                self.start[ng_indices] = (self.start[ng_indices] + 1) % self.num_cells
                self.steps_since_food[ng_indices] += 1

        mv_indices = torch.nonzero(moving, as_tuple=False).flatten()
        if mv_indices.numel() > 0:
            old_heads = head_pos[mv_indices]
            self.board[mv_indices, old_heads] = 2

            new_slot = (old_start[mv_indices] + old_length[mv_indices]) % self.num_cells
            new_head = next_pos[mv_indices]
            self.body[mv_indices, new_slot] = new_head
            self.occupancy[mv_indices, new_head] = True
            if self.occupancy_i64 is not None:
                self.occupancy_i64[mv_indices, new_head] = 1
            self.board[mv_indices, new_head] = self.head_codes[self.heading[mv_indices]]

        grow_indices = torch.nonzero(growing, as_tuple=False).flatten()
        if grow_indices.numel() > 0:
            if not self.cute_step_enabled:
                self.length[grow_indices] += 1
                self.steps_since_food[grow_indices] = 0
                rewards[grow_indices] += self.reward_food
            self.food[grow_indices] = -1
            still_running = grow_indices[self.length[grow_indices] < self.num_cells]
            if still_running.numel() > 0:
                if self.use_fast_cuda:
                    new_food = self._spawn_food_many(still_running)
                    valid = new_food >= 0
                    self.food[still_running] = new_food
                    if valid.any():
                        valid_indices = still_running[valid]
                        self.board[valid_indices, new_food[valid]] = 1
                else:
                    for index in still_running.tolist():
                        new_food = self._spawn_food(index)
                        self.food[index] = new_food
                        if new_food >= 0:
                            self.board[index, new_food] = 1

        if not self.cute_step_enabled:
            won = moving & (self.length == self.num_cells)
            truncated = moving & ~growing & (self.steps_since_food >= self.max_steps_since_food)
            done = dead | won | truncated
            self.done |= done
            self.episode_return += rewards
        else:
            done = self.done

        self.info_final_coverage.copy_(self.length)
        self.info_final_coverage.mul_(1.0 / float(self.num_cells))
        info = {
            "won": won,
            "truncated": truncated,
            "final_coverage": self.info_final_coverage,
            "episode_length": self.episode_step,
            "episode_return": self.episode_return,
        }
        return self.observe(), rewards, done, info

    def _spawn_food(self, index: int) -> int:
        occupied = self.occupancy[index].detach().cpu().tolist()
        empty = [cell for cell, used in enumerate(occupied) if not used]
        if not empty:
            return -1
        return self.rngs[index].choice(empty)

    def _spawn_food_many(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        empty = ~self.occupancy[indices]
        has_empty = empty.any(dim=1)
        if self.cuda_generator is not None:
            scores = torch.rand(
                (indices.numel(), self.num_cells),
                device=self.device,
                generator=self.cuda_generator,
            )
        else:
            scores = torch.rand((indices.numel(), self.num_cells), device=self.device)
        scores.masked_fill_(~empty, -1.0)
        food = scores.argmax(dim=1)
        return torch.where(has_empty, food, torch.full_like(food, -1))
