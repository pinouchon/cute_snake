from __future__ import annotations

import argparse
import logging
from pathlib import Path

from snake.config import apply_overrides, load_yaml_config, normalize_config, save_yaml_config
from snake.ppo import train_ppo
from snake.run_dirs import allocate_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = normalize_config(apply_overrides(load_yaml_config(args.config), args.overrides))
    run_dir = allocate_run_dir(config.get("run_root", "runs"), args.run_dir)
    save_yaml_config(config, run_dir / "config.yaml")

    logger = logging.getLogger("snake.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(run_dir / "train.log", encoding="utf-8"))

    result = train_ppo(config, Path(run_dir), logger)
    logger.info("training_complete=%s", result)


if __name__ == "__main__":
    main()
