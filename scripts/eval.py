from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from snake.config import load_yaml_config
from snake.eval import evaluate_policy
from snake.model import SnakePolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="latest.pt")
    parser.add_argument("--episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml_config(run_dir / "config.yaml")
    checkpoint = torch.load(run_dir / "checkpoints" / args.checkpoint, map_location="cpu")

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = SnakePolicy(
        board_size=int(config["board_size"]),
        hidden_size=int(config["hidden_size"]),
        model_type=str(config.get("model_type", "cnn")),
        transformer_layers=int(config.get("transformer_layers", 4)),
        transformer_heads=int(config.get("transformer_heads", 8)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    result = evaluate_policy(
        model,
        board_size=int(config["board_size"]),
        max_steps_since_food=int(config["max_steps_since_food"]),
        episodes=args.episodes or int(config["eval_episodes"]),
        seed=int(config["seed"]) + 20_000,
        device=device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
