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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    env = TorchSnakeBatchEnv(num_envs=args.num_envs, device=device)
    env.reset()

    start = time.perf_counter()
    for _ in range(args.steps):
        actions = torch.randint(0, 4, (args.num_envs,), device=env.device)
        _, _, dones, _ = env.step(actions)
        if torch.any(dones):
            env.reset(dones)
    elapsed = time.perf_counter() - start
    sps = (args.num_envs * args.steps) / elapsed
    print(f"device={device} num_envs={args.num_envs} steps={args.steps} sps={sps:.2f}")


if __name__ == "__main__":
    main()
