from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch


DIRECTION_OFFSETS = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

HEAD_CODES = {
    0: 3,
    1: 4,
    2: 5,
    3: 6,
}


@dataclass
class SnakeState:
    snake: list[int]
    heading: int
    food: int
    done: bool
    episode_step: int
    steps_since_food: int
    episode_return: float


class ReferenceSnakeEnv:
    def __init__(
        self,
        board_size: int = 8,
        max_steps_since_food: int = 128,
        seed: int = 0,
        initial_length: int = 3,
        reward_food: float = 1.0,
        reward_death: float = -1.0,
        reward_step: float = -0.01,
    ) -> None:
        self.board_size = board_size
        self.num_cells = board_size * board_size
        self.max_steps_since_food = max_steps_since_food
        self.seed = seed
        self.initial_length = initial_length
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.rng = random.Random(seed)
        self.state: SnakeState | None = None

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)
        row = self.board_size // 2
        snake = [row * self.board_size + column for column in range(1, 1 + self.initial_length)]
        self.state = SnakeState(
            snake=snake,
            heading=1,
            food=-1,
            done=False,
            episode_step=0,
            steps_since_food=0,
            episode_return=0.0,
        )
        self.state.food = self._spawn_food()
        return self.observation(), {}

    def force_state(
        self,
        snake: list[int],
        heading: int,
        food: int,
        *,
        done: bool = False,
        episode_step: int = 0,
        steps_since_food: int = 0,
        episode_return: float = 0.0,
    ) -> None:
        self.state = SnakeState(
            snake=list(snake),
            heading=heading,
            food=food,
            done=done,
            episode_step=episode_step,
            steps_since_food=steps_since_food,
            episode_return=episode_return,
        )

    def observation(self) -> torch.Tensor:
        state = self._require_state()
        board = torch.zeros(self.num_cells, dtype=torch.uint8)
        if state.food >= 0:
            board[state.food] = 1
        for cell in state.snake[:-1]:
            board[cell] = 2
        board[state.snake[-1]] = HEAD_CODES[state.heading]
        return board.view(self.board_size, self.board_size)

    def action_mask(self) -> torch.Tensor:
        state = self._require_state()
        mask = torch.ones(4, dtype=torch.bool)
        mask[(state.heading + 2) % 4] = False
        return mask

    def snapshot(self) -> dict[str, Any]:
        state = self._require_state()
        return {
            "snake": list(state.snake),
            "heading": state.heading,
            "food": state.food,
            "done": state.done,
            "episode_step": state.episode_step,
            "steps_since_food": state.steps_since_food,
            "episode_return": state.episode_return,
        }

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        state = self._require_state()
        if state.done:
            raise RuntimeError("Episode is done; call reset() before step().")

        if action == (state.heading + 2) % 4:
            action = state.heading
        state.heading = action
        state.episode_step += 1

        head = state.snake[-1]
        row, col = divmod(head, self.board_size)
        dr, dc = DIRECTION_OFFSETS[action]
        next_row = row + dr
        next_col = col + dc

        reward = self.reward_step
        info: dict[str, Any] = {
            "won": False,
            "truncated": False,
            "final_coverage": len(state.snake) / self.num_cells,
        }

        if not (0 <= next_row < self.board_size and 0 <= next_col < self.board_size):
            reward += self.reward_death
            state.done = True
            state.episode_return += reward
            info.update(self._terminal_info())
            return self.observation(), reward, True, info

        next_cell = next_row * self.board_size + next_col
        growing = next_cell == state.food
        tail = state.snake[0]
        occupied = set(state.snake)
        if next_cell in occupied and not (next_cell == tail and not growing):
            reward += self.reward_death
            state.done = True
            state.episode_return += reward
            info.update(self._terminal_info())
            return self.observation(), reward, True, info

        state.snake.append(next_cell)
        if growing:
            reward += self.reward_food
            state.steps_since_food = 0
            if len(state.snake) == self.num_cells:
                state.done = True
                info["won"] = True
            else:
                state.food = self._spawn_food()
        else:
            state.snake.pop(0)
            state.steps_since_food += 1

        if not state.done and state.steps_since_food >= self.max_steps_since_food:
            state.done = True
            info["truncated"] = True

        state.episode_return += reward
        info["final_coverage"] = len(state.snake) / self.num_cells
        if state.done:
            info.update(self._terminal_info())
        return self.observation(), reward, state.done, info

    def _terminal_info(self) -> dict[str, Any]:
        state = self._require_state()
        return {
            "episode_length": state.episode_step,
            "episode_return": state.episode_return,
            "final_coverage": len(state.snake) / self.num_cells,
        }

    def _spawn_food(self) -> int:
        state = self._require_state()
        occupied = set(state.snake)
        empty = [index for index in range(self.num_cells) if index not in occupied]
        if not empty:
            return -1
        return self.rng.choice(empty)

    def _require_state(self) -> SnakeState:
        if self.state is None:
            raise RuntimeError("Call reset() before using the environment.")
        return self.state
