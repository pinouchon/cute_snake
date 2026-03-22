from __future__ import annotations

import argparse
import time

import torch

from snake.env_gpu import TorchSnakeBatchEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-cute-step-core", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    env = TorchSnakeBatchEnv(
        num_envs=args.num_envs,
        device=device,
        use_cute_step_core=args.use_cute_step_core,
    )
    env.reset()

    start = time.perf_counter()
    for _ in range(args.steps):
        actions = torch.randint(0, 4, (args.num_envs,), device=env.device)
        _, _, dones, _ = env.step(actions)
        if bool(dones.any().item()):
            env.reset(dones)
    if env.device.type == "cuda":
        torch.cuda.synchronize(env.device)
    elapsed = time.perf_counter() - start
    sps = (args.num_envs * args.steps) / elapsed
    print(
        f"device={device} num_envs={args.num_envs} steps={args.steps} "
        f"use_cute_step_core={args.use_cute_step_core} sps={sps:.2f}"
    )


if __name__ == "__main__":
    main()
