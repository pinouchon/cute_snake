from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = [
        ["num_envs=1024", "rollout_steps=32", "learning_rate=0.0003"],
        ["num_envs=2048", "rollout_steps=32", "learning_rate=0.0003"],
    ]
    for run_overrides in overrides:
        command = ["uv", "run", "python", "scripts/train.py", "--config", args.config]
        for override in run_overrides:
            command.extend(["--set", override])
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
