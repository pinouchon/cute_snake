from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from snake.cute_kernels import HAVE_CUTE, CUTE_IMPORT_ERROR, step_core_cuda

try:
    import cutlass.cute as cute  # type: ignore
except Exception:  # pragma: no cover - depends on local install
    cute = None  # type: ignore

THREADS_PER_BLOCK = 256


@dataclass
class FusedStepExperimentState:
    actions: torch.Tensor
    heading: torch.Tensor
    head_pos: torch.Tensor
    tail_pos: torch.Tensor
    food: torch.Tensor
    done: torch.Tensor
    occupancy: torch.Tensor
    start: torch.Tensor
    length: torch.Tensor
    episode_step: torch.Tensor
    steps_since_food: torch.Tensor
    episode_return: torch.Tensor


def make_experiment_state(
    num_envs: int,
    board_size: int,
    *,
    device: str | torch.device,
    seed: int = 0,
) -> FusedStepExperimentState:
    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    num_cells = board_size * board_size
    actions = torch.randint(0, 4, (num_envs,), device=device, generator=generator, dtype=torch.int64)
    heading = torch.randint(0, 4, (num_envs,), device=device, generator=generator, dtype=torch.int64)
    start = torch.randint(0, num_cells, (num_envs,), device=device, generator=generator, dtype=torch.int64)
    length = torch.randint(3, min(num_cells, 10), (num_envs,), device=device, generator=generator, dtype=torch.int64)
    episode_step = torch.randint(0, 128, (num_envs,), device=device, generator=generator, dtype=torch.int64)
    steps_since_food = torch.randint(0, 128, (num_envs,), device=device, generator=generator, dtype=torch.int64)
    episode_return = torch.randn(num_envs, device=device, generator=generator, dtype=torch.float32)
    done = torch.zeros(num_envs, dtype=torch.int64, device=device)
    done[::11] = 1
    head_pos = torch.randint(0, num_cells, (num_envs,), device=device, generator=generator, dtype=torch.int64)
    tail_pos = torch.randint(0, num_cells, (num_envs,), device=device, generator=generator, dtype=torch.int64)
    food = torch.randint(-1, num_cells, (num_envs,), device=device, generator=generator, dtype=torch.int64)
    occupancy = torch.randint(0, 2, (num_envs, num_cells), device=device, generator=generator, dtype=torch.int64)
    return FusedStepExperimentState(
        actions=actions,
        heading=heading,
        head_pos=head_pos,
        tail_pos=tail_pos,
        food=food,
        done=done,
        occupancy=occupancy,
        start=start,
        length=length,
        episode_step=episode_step,
        steps_since_food=steps_since_food,
        episode_return=episode_return,
    )


def _torch_step_update_reference(
    *,
    actions: torch.Tensor,
    heading: torch.Tensor,
    head_pos: torch.Tensor,
    tail_pos: torch.Tensor,
    food: torch.Tensor,
    done: torch.Tensor,
    occupancy: torch.Tensor,
    start: torch.Tensor,
    length: torch.Tensor,
    episode_step: torch.Tensor,
    steps_since_food: torch.Tensor,
    episode_return: torch.Tensor,
    board_size: int,
    max_steps_since_food: int,
    reward_food: float,
    reward_death: float,
    reward_step: float,
) -> dict[str, torch.Tensor]:
    active = done == 0
    reverse = (heading + 2) % 4
    sanitized_actions = torch.where(actions == reverse, heading, actions)
    sanitized_actions = torch.where(active, sanitized_actions, heading)

    head_row = head_pos // board_size
    head_col = head_pos % board_size
    next_row = head_row + torch.tensor([-1, 0, 1, 0], device=head_pos.device)[sanitized_actions]
    next_col = head_col + torch.tensor([0, 1, 0, -1], device=head_pos.device)[sanitized_actions]
    wall = ((next_row < 0) | (next_row >= board_size) | (next_col < 0) | (next_col >= board_size)) & active
    next_row = next_row.clamp(0, board_size - 1)
    next_col = next_col.clamp(0, board_size - 1)
    next_pos = next_row * board_size + next_col

    growing = (next_pos == food) & active & ~wall
    occupied_next = occupancy.gather(1, next_pos.unsqueeze(1)).squeeze(1).to(torch.bool) & active & ~wall
    legal_tail = (next_pos == tail_pos) & ~growing
    dead = wall | (occupied_next & ~legal_tail)
    moving = active & ~dead

    reward = torch.full_like(episode_return, reward_step)
    reward = torch.where(active, reward, torch.zeros_like(reward))
    reward = torch.where(dead, reward + reward_death, reward)
    reward = torch.where(growing, reward + reward_food, reward)

    new_heading = torch.where(moving, sanitized_actions, heading)
    new_episode_step = episode_step + active.to(torch.int64)
    new_steps_since_food = torch.where(growing, torch.zeros_like(steps_since_food), steps_since_food + moving.to(torch.int64))
    new_length = length + growing.to(torch.int64)
    new_start = torch.where(moving & ~growing, (start + 1) % (board_size * board_size), start)
    won = moving & (new_length == board_size * board_size)
    truncated = moving & ~growing & (new_steps_since_food >= max_steps_since_food)
    new_done = done | dead.to(torch.int64) | won.to(torch.int64) | truncated.to(torch.int64)
    new_episode_return = episode_return + reward
    return {
        "sanitized_actions": sanitized_actions,
        "next_pos": next_pos,
        "wall": wall.to(torch.int64),
        "growing": growing.to(torch.int64),
        "dead": dead.to(torch.int64),
        "moving": moving.to(torch.int64),
        "new_heading": new_heading,
        "new_start": new_start,
        "new_length": new_length,
        "new_episode_step": new_episode_step,
        "new_steps_since_food": new_steps_since_food,
        "new_done": new_done,
        "new_episode_return": new_episode_return,
        "reward": reward,
        "won": won.to(torch.int64),
        "truncated": truncated.to(torch.int64),
    }


def _torch_bookkeeping_from_core(
    *,
    done: torch.Tensor,
    sanitized_actions: torch.Tensor,
    next_pos: torch.Tensor,
    wall: torch.Tensor,
    growing: torch.Tensor,
    dead: torch.Tensor,
    moving: torch.Tensor,
    heading: torch.Tensor,
    start: torch.Tensor,
    length: torch.Tensor,
    episode_step: torch.Tensor,
    steps_since_food: torch.Tensor,
    episode_return: torch.Tensor,
    board_size: int,
    max_steps_since_food: int,
    reward_food: float,
    reward_death: float,
    reward_step: float,
) -> dict[str, torch.Tensor]:
    active = done == 0
    reward = torch.full_like(episode_return, reward_step)
    reward = torch.where(active, reward, torch.zeros_like(reward))
    reward = torch.where(dead.to(torch.bool), reward + reward_death, reward)
    reward = torch.where(growing.to(torch.bool), reward + reward_food, reward)
    new_heading = torch.where(moving.to(torch.bool), sanitized_actions, heading)
    new_episode_step = episode_step + active.to(torch.int64)
    new_steps_since_food = torch.where(
        growing.to(torch.bool),
        torch.zeros_like(steps_since_food),
        steps_since_food + moving.to(torch.int64),
    )
    new_length = length + growing.to(torch.int64)
    new_start = torch.where(
        moving.to(torch.bool) & ~growing.to(torch.bool),
        (start + 1) % (board_size * board_size),
        start,
    )
    won = moving.to(torch.bool) & (new_length == board_size * board_size)
    truncated = moving.to(torch.bool) & ~growing.to(torch.bool) & (new_steps_since_food >= max_steps_since_food)
    new_done = dead | won.to(torch.int64) | truncated.to(torch.int64)
    new_episode_return = episode_return + reward
    return {
        "reward": reward,
        "new_heading": new_heading,
        "new_episode_step": new_episode_step,
        "new_steps_since_food": new_steps_since_food,
        "new_length": new_length,
        "new_start": new_start,
        "new_done": new_done,
        "new_episode_return": new_episode_return,
        "won": won.to(torch.int64),
        "truncated": truncated.to(torch.int64),
    }


if HAVE_CUTE:

    @cute.kernel
    def _device_fused_step_update(  # type: ignore[misc]
        actions: cute.Tensor,
        heading: cute.Tensor,
        head_pos: cute.Tensor,
        tail_pos: cute.Tensor,
        food: cute.Tensor,
        done: cute.Tensor,
        occupancy: cute.Tensor,
        start: cute.Tensor,
        length: cute.Tensor,
        episode_step: cute.Tensor,
        steps_since_food: cute.Tensor,
        episode_return: cute.Tensor,
        board_size: int,
        max_steps_since_food: int,
        reward_food: float,
        reward_death: float,
        reward_step: float,
        sanitized_actions: cute.Tensor,
        next_pos: cute.Tensor,
        wall: cute.Tensor,
        growing: cute.Tensor,
        dead: cute.Tensor,
        moving: cute.Tensor,
        new_heading: cute.Tensor,
        new_start: cute.Tensor,
        new_length: cute.Tensor,
        new_episode_step: cute.Tensor,
        new_steps_since_food: cute.Tensor,
        new_done: cute.Tensor,
        new_episode_return: cute.Tensor,
        reward: cute.Tensor,
        won: cute.Tensor,
        truncated: cute.Tensor,
    ):
        block_x, _, _ = cute.arch.block_idx()
        tid_x, _, _ = cute.arch.thread_idx()
        tid = block_x * THREADS_PER_BLOCK + tid_x
        if tid < actions.shape[0]:
            active = 0 if done[tid] != 0 else 1
            current_heading = heading[tid]
            action = actions[tid]
            reverse = (current_heading + 2) % 4
            if action == reverse:
                action = current_heading
            if active == 0:
                action = current_heading
            sanitized_actions[tid] = action
            zero = done[tid] * 0
            one = zero + 1

            head = head_pos[tid]
            row = head // board_size
            col = head - row * board_size
            if action == 0:
                row = row - 1
            elif action == 1:
                col = col + 1
            elif action == 2:
                row = row + 1
            else:
                col = col - 1

            is_wall = False
            if row < 0:
                is_wall = True
            if row >= board_size:
                is_wall = True
            if col < 0:
                is_wall = True
            if col >= board_size:
                is_wall = True

            local_reward = episode_return[tid] * 0.0
            if active == 1:
                local_reward = reward_step
            growing_flag = done[tid] * 0
            dead_flag = done[tid] * 0
            moving_flag = done[tid] * 0
            next_cell = head
            if is_wall:
                if active == 1:
                    dead_flag = one
                if active == 1:
                    local_reward = local_reward + reward_death
            else:
                next_cell = row * board_size + col
                if active == 1 and next_cell == food[tid]:
                    growing_flag = one
                occupied_next = 0
                if active == 1:
                    occupied_next = 1 if occupancy[tid, next_cell] != 0 else 0
                legal_tail = 1 if (growing_flag == 0 and next_cell == tail_pos[tid]) else 0
                if occupied_next == 1 and legal_tail == 0:
                    dead_flag = one
                else:
                    if active == 1:
                        moving_flag = one
                if growing_flag == 1:
                    local_reward = local_reward + reward_food

            if is_wall:
                wall[tid] = one
            else:
                wall[tid] = zero
            growing[tid] = growing_flag
            dead[tid] = dead_flag
            moving[tid] = moving_flag
            next_pos[tid] = next_cell
            if moving_flag == 1:
                new_heading[tid] = action
            else:
                new_heading[tid] = current_heading
            new_episode_step[tid] = episode_step[tid] + active
            new_length_value = length[tid] + growing_flag
            new_length[tid] = new_length_value
            if moving_flag == 1 and growing_flag == 0:
                new_start[tid] = (start[tid] + 1) % (board_size * board_size)
            else:
                new_start[tid] = start[tid]
            new_steps_since_food_value = steps_since_food[tid]
            if growing_flag == 1:
                new_steps_since_food_value = zero
            else:
                if moving_flag == 1:
                    new_steps_since_food_value = steps_since_food[tid] + 1
            new_steps_since_food[tid] = new_steps_since_food_value
            won_flag = zero
            if moving_flag == 1 and new_length_value == board_size * board_size:
                won_flag = one
            truncated_flag = zero
            if moving_flag == 1 and growing_flag == 0 and new_steps_since_food_value >= max_steps_since_food:
                truncated_flag = one
            won[tid] = won_flag
            truncated[tid] = truncated_flag
            done_next = done[tid] | dead_flag | won_flag | truncated_flag
            new_done[tid] = done_next
            new_episode_return[tid] = episode_return[tid] + local_reward
            reward[tid] = local_reward

    @cute.jit
    def _fused_step_update(
        actions: cute.Tensor,
        heading: cute.Tensor,
        head_pos: cute.Tensor,
        tail_pos: cute.Tensor,
        food: cute.Tensor,
        done: cute.Tensor,
        occupancy: cute.Tensor,
        start: cute.Tensor,
        length: cute.Tensor,
        episode_step: cute.Tensor,
        steps_since_food: cute.Tensor,
        episode_return: cute.Tensor,
        board_size: int,
        max_steps_since_food: int,
        reward_food: float,
        reward_death: float,
        reward_step: float,
        sanitized_actions: cute.Tensor,
        next_pos: cute.Tensor,
        wall: cute.Tensor,
        growing: cute.Tensor,
        dead: cute.Tensor,
        moving: cute.Tensor,
        new_heading: cute.Tensor,
        new_start: cute.Tensor,
        new_length: cute.Tensor,
        new_episode_step: cute.Tensor,
        new_steps_since_food: cute.Tensor,
        new_done: cute.Tensor,
        new_episode_return: cute.Tensor,
        reward: cute.Tensor,
        won: cute.Tensor,
        truncated: cute.Tensor,
    ):
        blocks = (actions.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        _device_fused_step_update(
            actions,
            heading,
            head_pos,
            tail_pos,
            food,
            done,
            occupancy,
            start,
            length,
            episode_step,
            steps_since_food,
            episode_return,
            board_size,
            max_steps_since_food,
            reward_food,
            reward_death,
            reward_step,
            sanitized_actions,
            next_pos,
            wall,
            growing,
            dead,
            moving,
            new_heading,
            new_start,
            new_length,
            new_episode_step,
            new_steps_since_food,
            new_done,
            new_episode_return,
            reward,
            won,
            truncated,
        ).launch(
            grid=(blocks, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
        )


@lru_cache(maxsize=4)
def _compiled_fused_step_update(board_size: int, max_steps_since_food: int):
    if not HAVE_CUTE:
        raise RuntimeError(f"CuTe DSL is unavailable: {CUTE_IMPORT_ERROR}")
    n = cute.sym_int()
    vec_i64 = lambda: cute.runtime.make_fake_compact_tensor(cute.Int64, (n,))
    vec_f32 = lambda: cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
    occupancy = cute.runtime.make_fake_compact_tensor(
        cute.Int64,
        (n, board_size * board_size),
        stride_order=(1, 0),
    )
    return cute.compile(
        _fused_step_update,
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        occupancy,
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_f32(),
        board_size,
        max_steps_since_food,
        1.0,
        -1.0,
        -0.01,
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_f32(),
        vec_f32(),
        vec_i64(),
        vec_i64(),
    )


def fused_step_update_cuda(
    *,
    actions: torch.Tensor,
    heading: torch.Tensor,
    head_pos: torch.Tensor,
    tail_pos: torch.Tensor,
    food: torch.Tensor,
    done: torch.Tensor,
    occupancy: torch.Tensor,
    start: torch.Tensor,
    length: torch.Tensor,
    episode_step: torch.Tensor,
    steps_since_food: torch.Tensor,
    episode_return: torch.Tensor,
    board_size: int,
    max_steps_since_food: int,
    reward_food: float,
    reward_death: float,
    reward_step: float,
    sanitized_actions: torch.Tensor,
    next_pos: torch.Tensor,
    wall: torch.Tensor,
    growing: torch.Tensor,
    dead: torch.Tensor,
    moving: torch.Tensor,
    new_heading: torch.Tensor,
    new_start: torch.Tensor,
    new_length: torch.Tensor,
    new_episode_step: torch.Tensor,
    new_steps_since_food: torch.Tensor,
    new_done: torch.Tensor,
    new_episode_return: torch.Tensor,
    reward: torch.Tensor,
    won: torch.Tensor,
    truncated: torch.Tensor,
) -> None:
    if not HAVE_CUTE:
        raise RuntimeError(f"CuTe DSL is unavailable: {CUTE_IMPORT_ERROR}")
    compiled = _compiled_fused_step_update(board_size, max_steps_since_food)
    compiled(
        actions,
        heading,
        head_pos,
        tail_pos,
        food,
        done,
        occupancy,
        start,
        length,
        episode_step,
        steps_since_food,
        episode_return,
        board_size,
        max_steps_since_food,
        reward_food,
        reward_death,
        reward_step,
        sanitized_actions,
        next_pos,
        wall,
        growing,
        dead,
        moving,
        new_heading,
        new_start,
        new_length,
        new_episode_step,
        new_steps_since_food,
        new_done,
        new_episode_return,
        reward,
        won,
        truncated,
    )


def benchmark_fused_step_update(
    num_envs: int,
    board_size: int = 8,
    steps: int = 500,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> dict[str, Any]:
    device = torch.device(device)
    state = make_experiment_state(num_envs, board_size, device=device, seed=seed)
    num_cells = board_size * board_size

    sanitized_actions = torch.empty_like(state.actions)
    next_pos = torch.empty_like(state.actions)
    wall = torch.empty_like(state.actions)
    growing = torch.empty_like(state.actions)
    dead = torch.empty_like(state.actions)
    moving = torch.empty_like(state.actions)
    new_heading = torch.empty_like(state.actions)
    new_start = torch.empty_like(state.actions)
    new_length = torch.empty_like(state.actions)
    new_episode_step = torch.empty_like(state.actions)
    new_steps_since_food = torch.empty_like(state.actions)
    new_done = torch.empty_like(state.actions)
    new_episode_return = torch.empty_like(state.episode_return)
    reward = torch.empty_like(state.episode_return)
    won = torch.empty_like(state.actions)
    truncated = torch.empty_like(state.actions)
    core_wall = torch.empty_like(state.actions)
    core_growing = torch.empty_like(state.actions)
    core_dead = torch.empty_like(state.actions)
    core_moving = torch.empty_like(state.actions)

    def baseline_step() -> None:
        step_core_cuda(
            actions=state.actions,
            heading=state.heading,
            head_pos=state.head_pos,
            tail_pos=state.tail_pos,
            food=state.food,
            done=state.done,
            occupancy=state.occupancy,
            board_size=board_size,
            sanitized_actions=sanitized_actions,
            next_pos=next_pos,
            wall=core_wall,
            growing=core_growing,
            dead=core_dead,
            moving=core_moving,
        )
        out = _torch_bookkeeping_from_core(
            done=state.done,
            sanitized_actions=sanitized_actions,
            next_pos=next_pos,
            wall=core_wall,
            growing=core_growing,
            dead=core_dead,
            moving=core_moving,
            heading=state.heading,
            start=state.start,
            length=state.length,
            episode_step=state.episode_step,
            steps_since_food=state.steps_since_food,
            episode_return=state.episode_return,
            board_size=board_size,
            max_steps_since_food=128,
            reward_food=1.0,
            reward_death=-1.0,
            reward_step=-0.01,
        )
        _ = out["reward"].sum()

    def fused_step() -> None:
        fused_step_update_cuda(
            actions=state.actions,
            heading=state.heading,
            head_pos=state.head_pos,
            tail_pos=state.tail_pos,
            food=state.food,
            done=state.done,
            occupancy=state.occupancy,
            start=state.start,
            length=state.length,
            episode_step=state.episode_step,
            steps_since_food=state.steps_since_food,
            episode_return=state.episode_return,
            board_size=board_size,
            max_steps_since_food=128,
            reward_food=1.0,
            reward_death=-1.0,
            reward_step=-0.01,
            sanitized_actions=sanitized_actions,
            next_pos=next_pos,
            wall=wall,
            growing=growing,
            dead=dead,
            moving=moving,
            new_heading=new_heading,
            new_start=new_start,
            new_length=new_length,
            new_episode_step=new_episode_step,
            new_steps_since_food=new_steps_since_food,
            new_done=new_done,
            new_episode_return=new_episode_return,
            reward=reward,
            won=won,
            truncated=truncated,
        )

    def time_it(fn) -> float:
        for _ in range(10):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        import time

        wall_start = time.perf_counter()
        for _ in range(steps):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter() - wall_start

    baseline_seconds = time_it(baseline_step)
    fused_seconds = time_it(fused_step)
    return {
        "num_envs": num_envs,
        "board_size": board_size,
        "steps": steps,
        "baseline_seconds": baseline_seconds,
        "fused_seconds": fused_seconds,
        "baseline_steps_per_second": (num_envs * steps) / baseline_seconds,
        "fused_steps_per_second": (num_envs * steps) / fused_seconds,
    }
