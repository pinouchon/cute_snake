from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

try:
    import cutlass.cute as cute  # type: ignore

    HAVE_CUTE = True
    CUTE_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - depends on local install
    cute = None  # type: ignore
    HAVE_CUTE = False
    CUTE_IMPORT_ERROR = str(exc)

THREADS_PER_BLOCK = 256


if HAVE_CUTE:

    @cute.kernel
    def _device_add_one(src: cute.Tensor, dst: cute.Tensor):  # type: ignore[misc]
        block_x, _, _ = cute.arch.block_idx()
        tid_x, _, _ = cute.arch.thread_idx()
        tid = block_x * THREADS_PER_BLOCK + tid_x
        if tid < src.shape[0]:
            dst[tid] = src[tid] + 1.0

    @cute.jit
    def _add_one(src: cute.Tensor, dst: cute.Tensor):
        blocks = (src.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        _device_add_one(src, dst).launch(
            grid=(blocks, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
        )

    @cute.kernel
    def _device_step_core(  # type: ignore[misc]
        actions: cute.Tensor,
        heading: cute.Tensor,
        head_pos: cute.Tensor,
        tail_pos: cute.Tensor,
        food: cute.Tensor,
        done: cute.Tensor,
        occupancy: cute.Tensor,
        board_size: int,
        sanitized_actions: cute.Tensor,
        next_pos: cute.Tensor,
        wall: cute.Tensor,
        growing: cute.Tensor,
        dead: cute.Tensor,
        moving: cute.Tensor,
    ):
        block_x, _, _ = cute.arch.block_idx()
        tid_x, _, _ = cute.arch.thread_idx()
        tid = block_x * THREADS_PER_BLOCK + tid_x
        if tid < actions.shape[0]:
            current_heading = heading[tid]
            action = actions[tid]
            reverse = (current_heading + 2) % 4
            if action == reverse:
                action = current_heading
            if done[tid] != 0:
                action = current_heading
            sanitized_actions[tid] = action

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

            if is_wall:
                wall[tid] = 1
                growing[tid] = 0
                dead[tid] = 1 - done[tid]
                moving[tid] = 0
                next_pos[tid] = head
            else:
                cell = row * board_size + col
                is_growing = False
                if done[tid] == 0 and cell == food[tid]:
                    is_growing = True

                occupied_next = False
                if done[tid] == 0:
                    occupied_next = occupancy[tid, cell] != 0

                legal_tail = False
                if not is_growing and cell == tail_pos[tid]:
                    legal_tail = True

                is_dead = occupied_next and not legal_tail

                wall[tid] = 0
                growing[tid] = 1 if is_growing else 0
                dead[tid] = 1 if is_dead else 0
                moving[tid] = 1 if (done[tid] == 0 and not is_dead) else 0
                next_pos[tid] = cell

    @cute.jit
    def _step_core(
        actions: cute.Tensor,
        heading: cute.Tensor,
        head_pos: cute.Tensor,
        tail_pos: cute.Tensor,
        food: cute.Tensor,
        done: cute.Tensor,
        occupancy: cute.Tensor,
        board_size: int,
        sanitized_actions: cute.Tensor,
        next_pos: cute.Tensor,
        wall: cute.Tensor,
        growing: cute.Tensor,
        dead: cute.Tensor,
        moving: cute.Tensor,
    ):
        blocks = (actions.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        _device_step_core(
            actions,
            heading,
            head_pos,
            tail_pos,
            food,
            done,
            occupancy,
            board_size,
            sanitized_actions,
            next_pos,
            wall,
            growing,
            dead,
            moving,
        ).launch(
            grid=(blocks, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
        )


@lru_cache(maxsize=1)
def _compiled_add_one():
    if not HAVE_CUTE:
        raise RuntimeError(f"CuTe DSL is unavailable: {CUTE_IMPORT_ERROR}")
    n = cute.sym_int()
    src = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
    dst = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
    return cute.compile(_add_one, src, dst)


@lru_cache(maxsize=4)
def _compiled_step_core(board_size: int):
    if not HAVE_CUTE:
        raise RuntimeError(f"CuTe DSL is unavailable: {CUTE_IMPORT_ERROR}")
    n = cute.sym_int()
    vec_i64 = lambda: cute.runtime.make_fake_compact_tensor(cute.Int64, (n,))
    occupancy = cute.runtime.make_fake_compact_tensor(
        cute.Int64,
        (n, board_size * board_size),
        stride_order=(1, 0),
    )
    return cute.compile(
        _step_core,
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        occupancy,
        board_size,
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
        vec_i64(),
    )


def add_one_cuda_tensor(src: torch.Tensor) -> torch.Tensor:
    """Add one to a 1D CUDA tensor via a CuTe kernel."""

    if not HAVE_CUTE:
        raise RuntimeError(f"CuTe DSL is unavailable: {CUTE_IMPORT_ERROR}")
    if src.device.type != "cuda":
        raise ValueError("add_one_cuda_tensor expects a CUDA tensor")
    if src.dtype != torch.float32:
        raise TypeError("add_one_cuda_tensor expects float32 input")
    if src.dim() != 1:
        raise ValueError("add_one_cuda_tensor expects a 1D tensor")

    dst = torch.empty_like(src)
    compiled = _compiled_add_one()
    compiled(src, dst)
    torch.cuda.synchronize(src.device)
    return dst


def run_cute_smoke() -> torch.Tensor:
    """Run a minimal CuTe kernel against torch CUDA tensors."""

    src = torch.arange(16, dtype=torch.float32, device="cuda")
    return add_one_cuda_tensor(src)


def step_core_cuda(
    *,
    actions: torch.Tensor,
    heading: torch.Tensor,
    head_pos: torch.Tensor,
    tail_pos: torch.Tensor,
    food: torch.Tensor,
    done: torch.Tensor,
    occupancy: torch.Tensor,
    board_size: int,
    sanitized_actions: torch.Tensor,
    next_pos: torch.Tensor,
    wall: torch.Tensor,
    growing: torch.Tensor,
    dead: torch.Tensor,
    moving: torch.Tensor,
) -> None:
    if not HAVE_CUTE:
        raise RuntimeError(f"CuTe DSL is unavailable: {CUTE_IMPORT_ERROR}")
    compiled = _compiled_step_core(board_size)
    compiled(
        actions,
        heading,
        head_pos,
        tail_pos,
        food,
        done,
        occupancy,
        board_size,
        sanitized_actions,
        next_pos,
        wall,
        growing,
        dead,
        moving,
    )


def capability_summary() -> dict[str, Any]:
    return {
        "available": HAVE_CUTE,
        "import_error": CUTE_IMPORT_ERROR,
        "module": "cutlass.cute",
    }
