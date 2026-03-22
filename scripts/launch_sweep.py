from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep_wallclock.yaml")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_sweep_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping config at {path}, got {type(data)!r}")
    return data


def _load_base_only(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        yaml.safe_load(handle)
    return {"base": str(path), "sweep": {}}


def _expand_grid(sweep: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = sorted(sweep)
    if not keys:
        return [{}]
    values = [sweep[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _format_override(value: Any) -> str:
    return json.dumps(value)


def _preview_next_run_dir(root: str | Path) -> Path:
    root_path = Path(root)
    existing = [
        int(path.name)
        for path in root_path.iterdir()
        if path.is_dir() and path.name.isdigit()
    ] if root_path.exists() else []
    return root_path / f"{(max(existing, default=0) + 1):04d}"


def _plan_run_dirs(root: str | Path, count: int) -> list[Path]:
    root_path = Path(root)
    existing = [
        int(path.name)
        for path in root_path.iterdir()
        if path.is_dir() and path.name.isdigit()
    ] if root_path.exists() else []
    start = max(existing, default=0) + 1
    return [root_path / f"{start + offset:04d}" for offset in range(count)]


def _launch_output_relay(prefix: str, stream) -> threading.Thread:
    def _relay() -> None:
        for line in iter(stream.readline, ""):
            sys.stdout.write(f"{prefix}{line}")
            sys.stdout.flush()

    thread = threading.Thread(target=_relay, daemon=True)
    thread.start()
    return thread


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / args.config
    sweep_config = _load_sweep_config(config_path) if config_path.name != "base.yaml" else _load_base_only(config_path)
    base_config = sweep_config.get("base", "configs/base.yaml")
    sweep = sweep_config.get("sweep", {})
    if not isinstance(sweep, dict):
        raise TypeError("Expected sweep to be a mapping")

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("No GPUs specified")

    configs = _expand_grid({key: list(value) for key, value in sweep.items()})
    start = max(0, int(args.offset))
    stop = start + (args.limit if args.limit is not None else len(gpus))
    selected = configs[start:stop]
    if not selected:
        raise ValueError("No configurations selected")

    planned_run_dirs = _plan_run_dirs(args.runs_root, len(selected))
    launched: list[dict[str, Any]] = []
    for index, overrides in enumerate(selected):
        gpu = gpus[index % len(gpus)]
        run_dir = planned_run_dirs[index]
        command = [
            "uv",
            "run",
            "python",
            str(repo_root / "scripts" / "train.py"),
            "--config",
            str(base_config),
            "--run-dir",
            str(run_dir),
        ]
        for key in sorted(overrides):
            command.extend(["--set", f"{key}={_format_override(overrides[key])}"])

        record = {
            "gpu": gpu,
            "run_dir": str(run_dir),
            "command": command,
            "overrides": overrides,
        }
        launched.append(record)

        if args.dry_run:
            print(json.dumps(record, indent=2, sort_keys=True))
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=repo_root,
        )
        record["process"] = proc
        prefix = f"[gpu={gpu} run={run_dir.name}] "
        assert proc.stdout is not None
        record["relay"] = _launch_output_relay(prefix, proc.stdout)
        print(f"launched gpu={gpu} run={run_dir} overrides={overrides}")

    if args.dry_run:
        return

    exit_codes: list[int] = []
    for record in launched:
        proc = record["process"]
        timeout_seconds = float(args.timeout_seconds)
        if timeout_seconds > 0.0:
            deadline = time.time() + timeout_seconds
            exit_code: int | None = None
            while exit_code is None and time.time() < deadline:
                exit_code = proc.poll()
                if exit_code is None:
                    time.sleep(0.2)
            if exit_code is None:
                proc.terminate()
                try:
                    exit_code = proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    exit_code = proc.wait()
                print(f"timed_out run={record['run_dir']} gpu={record['gpu']} exit_code={exit_code}")
        else:
            exit_code = proc.wait()
        exit_codes.append(exit_code)
        relay = record["relay"]
        relay.join(timeout=1.0)
        status = "ok" if exit_code == 0 else "failed"
        print(f"completed run={record['run_dir']} gpu={record['gpu']} status={status} exit_code={exit_code}")

    if any(code != 0 for code in exit_codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
