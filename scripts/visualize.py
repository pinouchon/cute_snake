from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snake.config import load_yaml_config
from snake.env_reference import ReferenceSnakeEnv
from snake.model import SnakePolicy


CELL_GLYPHS = {
    0: " .",
    1: " *",
    2: " o",
    3: " ^",
    4: " >",
    5: " v",
    6: " <",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="latest.pt")
    parser.add_argument("--fps", type=float, default=6.0)
    return parser.parse_args()


def render_frame(obs: torch.Tensor) -> str:
    board = obs.cpu().tolist()
    rows = []
    for row in board:
        rows.append("".join(CELL_GLYPHS[int(cell)] for cell in row))
    return "\n".join(rows)


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
    model.eval()

    env = ReferenceSnakeEnv(
        board_size=int(config["board_size"]),
        max_steps_since_food=int(config["max_steps_since_food"]),
        seed=int(config["seed"]) + 30_000,
        initial_length=int(config["initial_length"]),
        reward_food=float(config["reward_food"]),
        reward_death=float(config["reward_death"]),
        reward_step=float(config["reward_step"]),
    )
    obs, _ = env.reset(seed=int(config["seed"]) + 30_000)
    done = False
    frame_delay = 1.0 / max(args.fps, 0.1)
    print("\x1b[2J", end="")
    while not done:
        print("\x1b[H", end="")
        print(render_frame(obs))
        snapshot = env.snapshot()
        print(
            f"\nstep={snapshot['episode_step']} return={snapshot['episode_return']:.2f} "
            f"length={len(snapshot['snake'])} food={snapshot['food']}"
        )
        time.sleep(frame_delay)
        obs_tensor = obs.unsqueeze(0).to(device)
        mask = env.action_mask().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(obs_tensor)
            action = torch.argmax(
                logits.masked_fill(~mask, torch.finfo(logits.dtype).min),
                dim=-1,
            )
        obs, _, done, _ = env.step(int(action.item()))
    print("\x1b[H", end="")
    print(render_frame(obs))
    snapshot = env.snapshot()
    print(
        f"\nfinal_step={snapshot['episode_step']} return={snapshot['episode_return']:.2f} "
        f"length={len(snapshot['snake'])} done={snapshot['done']}"
    )
    if os.getenv("TERM") is None:
        print("warning: TERM is not set; terminal redraw may be limited.")


if __name__ == "__main__":
    main()
