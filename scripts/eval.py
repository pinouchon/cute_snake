from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from snake.config import load_yaml_config
from snake.eval import evaluate_policy
from snake.implementations.implementation4 import build_policy
from snake.model import load_policy_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="latest.pt")
    parser.add_argument("--episodes", type=int, default=None)
    return parser.parse_args()


def resolve_checkpoint_path(run_dir: Path, checkpoint_name: str) -> Path:
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    if checkpoint_path.exists():
        return checkpoint_path
    if checkpoint_name == "latest.pt":
        fallback = run_dir / "checkpoints" / "best.pt"
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml_config(run_dir / "config.yaml")
    checkpoint = torch.load(resolve_checkpoint_path(run_dir, args.checkpoint), map_location="cpu")

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = build_policy(config)
    load_policy_state(model, checkpoint["model"])
    model.to(device)

    result = evaluate_policy(
        model,
        board_size=int(config["board_size"]),
        max_steps_since_food=int(config["max_steps_since_food"]),
        episodes=args.episodes or int(config["eval_episodes"]),
        seed=int(config["seed"]) + 10_000,
        device=device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
