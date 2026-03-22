from __future__ import annotations

import argparse
import json

from snake.cute_fused_experiment import benchmark_fused_step_update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--board-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark_fused_step_update(
        num_envs=args.num_envs,
        board_size=args.board_size,
        steps=args.steps,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
