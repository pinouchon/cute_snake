from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


SEARCH_ROOT_KEYS = (
    "learning_rate",
    "clip_coef",
    "entropy_coef",
    "gae_lambda",
    "use_value_clipping",
    "value_coef",
    "max_grad_norm",
    "trunk_channels",
    "hidden_size",
)

SEARCH_SPACE: dict[str, list[Any]] = {
    "learning_rate": [0.0026, 0.0027, 0.00275, 0.0028, 0.00285, 0.0029, 0.0030, 0.0031],
    "clip_coef": [0.18, 0.19, 0.2, 0.21, 0.22, 0.24],
    "entropy_coef": [0.003, 0.004, 0.005, 0.006],
    "gae_lambda": [0.975, 0.98, 0.985],
    "use_value_clipping": [True, False],
    "value_coef": [0.4, 0.5, 0.6],
    "max_grad_norm": [0.4, 0.5, 0.6, 0.7],
    "model_preset": [
        {"trunk_channels": [20, 40], "hidden_size": 160},
        {"trunk_channels": [24, 48], "hidden_size": 192},
        {"trunk_channels": [28, 56], "hidden_size": 224},
    ],
}

FIXED_OVERRIDES = [
    "implementation=implementation4",
    "num_envs=4096",
    "rollout_steps=16",
    "minibatches=8",
    "update_epochs=2",
    "compile_model=false",
    "use_cute_step_core=false",
    "profile_interval_updates=0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/implementation4.yaml")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--state-dir", default="artifacts/impl4_round_search")
    parser.add_argument("--seed", type=int, default=106)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping config at {path}, got {type(data)!r}")
    return data


def _iter_run_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.isdigit())


def _is_current_impl4_family(config: dict[str, Any]) -> bool:
    return (
        config.get("implementation") == "implementation4"
        and int(config.get("num_envs", -1)) == 4096
        and int(config.get("rollout_steps", -1)) == 16
        and int(config.get("minibatches", -1)) == 8
        and int(config.get("update_epochs", -1)) == 2
        and config.get("model_type", "cnn") == "cnn"
    )


def _run_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    trunk = tuple(int(value) for value in config.get("trunk_channels", [24, 48]))
    return (
        round(float(config.get("learning_rate", 0.0)), 6),
        round(float(config.get("clip_coef", 0.0)), 4),
        round(float(config.get("entropy_coef", 0.0)), 4),
        round(float(config.get("gae_lambda", 0.0)), 4),
        bool(config.get("use_value_clipping", True)),
        round(float(config.get("value_coef", 0.0)), 4),
        round(float(config.get("max_grad_norm", 0.0)), 4),
        trunk,
        int(config.get("hidden_size", 0)),
    )


def _summarize_run(run_dir: Path, selection_target: float = 0.80) -> dict[str, Any] | None:
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    if not config_path.exists() or not metrics_path.exists():
        return None
    config = _load_yaml(config_path)
    if not _is_current_impl4_family(config):
        return None
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = [json.loads(line) for line in handle if line.strip()]
    if not metrics:
        return None
    best_eval = max((record.get("eval_mean_final_coverage", -1.0) for record in metrics), default=-1.0)
    success = next((record for record in metrics if record.get("eval_mean_final_coverage", 0.0) >= selection_target), None)
    return {
        "run_dir": run_dir.name,
        "config": config,
        "signature": _run_signature(config),
        "success": success is not None,
        "success_elapsed_seconds": None if success is None else float(success["elapsed_seconds"]),
        "success_update": None if success is None else int(success["update"]),
        "success_coverage": None if success is None else float(success["eval_mean_final_coverage"]),
        "best_eval_coverage": float(best_eval),
        "final_elapsed_seconds": float(metrics[-1]["elapsed_seconds"]),
    }


def _rank_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    if summary["success"]:
        return (0, float(summary["success_elapsed_seconds"]), -float(summary["success_coverage"]), summary["run_dir"])
    return (1, -float(summary["best_eval_coverage"]), float(summary["final_elapsed_seconds"]), summary["run_dir"])


def _load_history(runs_root: Path) -> list[dict[str, Any]]:
    summaries = []
    for run_dir in _iter_run_dirs(runs_root):
        summary = _summarize_run(run_dir)
        if summary is not None:
            summaries.append(summary)
    summaries.sort(key=_rank_key)
    return summaries


def _base_candidate(best_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "learning_rate": float(best_config["learning_rate"]),
        "clip_coef": float(best_config["clip_coef"]),
        "entropy_coef": float(best_config["entropy_coef"]),
        "gae_lambda": float(best_config["gae_lambda"]),
        "use_value_clipping": bool(best_config["use_value_clipping"]),
        "value_coef": float(best_config["value_coef"]),
        "max_grad_norm": float(best_config["max_grad_norm"]),
        "trunk_channels": [int(value) for value in best_config["trunk_channels"]],
        "hidden_size": int(best_config["hidden_size"]),
    }


def _neighbor(values: list[Any], current: Any, rng: random.Random) -> Any:
    if current not in values:
        return rng.choice(values)
    index = values.index(current)
    candidates = [index]
    if index > 0:
        candidates.append(index - 1)
    if index + 1 < len(values):
        candidates.append(index + 1)
    chosen = rng.choice(candidates)
    return values[chosen]


def _mutate_candidate(candidate: dict[str, Any], rng: random.Random, *, intensity: int) -> dict[str, Any]:
    mutated = {
        "learning_rate": float(candidate["learning_rate"]),
        "clip_coef": float(candidate["clip_coef"]),
        "entropy_coef": float(candidate["entropy_coef"]),
        "gae_lambda": float(candidate["gae_lambda"]),
        "use_value_clipping": bool(candidate["use_value_clipping"]),
        "value_coef": float(candidate["value_coef"]),
        "max_grad_norm": float(candidate["max_grad_norm"]),
        "trunk_channels": list(candidate["trunk_channels"]),
        "hidden_size": int(candidate["hidden_size"]),
    }
    mutation_keys = [
        "learning_rate",
        "clip_coef",
        "entropy_coef",
        "gae_lambda",
        "use_value_clipping",
        "value_coef",
        "max_grad_norm",
        "model_preset",
    ]
    for _ in range(intensity):
        key = rng.choice(mutation_keys)
        if key == "model_preset":
            preset = _neighbor(
                SEARCH_SPACE["model_preset"],
                {"trunk_channels": mutated["trunk_channels"], "hidden_size": mutated["hidden_size"]},
                rng,
            )
            mutated["trunk_channels"] = list(preset["trunk_channels"])
            mutated["hidden_size"] = int(preset["hidden_size"])
            continue
        if key == "use_value_clipping":
            mutated[key] = not bool(mutated[key])
            continue
        values = SEARCH_SPACE[key]
        mutated[key] = _neighbor(values, mutated[key], rng)
    return mutated


def _candidate_signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        round(float(candidate["learning_rate"]), 6),
        round(float(candidate["clip_coef"]), 4),
        round(float(candidate["entropy_coef"]), 4),
        round(float(candidate["gae_lambda"]), 4),
        bool(candidate["use_value_clipping"]),
        round(float(candidate["value_coef"]), 4),
        round(float(candidate["max_grad_norm"]), 4),
        tuple(int(value) for value in candidate["trunk_channels"]),
        int(candidate["hidden_size"]),
    )


def _candidate_to_overrides(candidate: dict[str, Any], round_index: int, slot: int, seed: int) -> list[str]:
    overrides = list(FIXED_OVERRIDES)
    overrides.extend(
        [
            f"seed={seed}",
            f"learning_rate={candidate['learning_rate']}",
            f"clip_coef={candidate['clip_coef']}",
            f"entropy_coef={candidate['entropy_coef']}",
            f"gae_lambda={candidate['gae_lambda']}",
            f"use_value_clipping={str(candidate['use_value_clipping']).lower()}",
            f"value_coef={candidate['value_coef']}",
            f"max_grad_norm={candidate['max_grad_norm']}",
            f"trunk_channels={json.dumps(candidate['trunk_channels'])}",
            f"hidden_size={candidate['hidden_size']}",
        ]
    )
    return overrides


def _plan_round(history: list[dict[str, Any]], round_index: int, rng: random.Random) -> list[dict[str, Any]]:
    ordered = sorted(history, key=_rank_key)
    best = ordered[0]["config"]
    parents = [entry["config"] for entry in ordered[: min(6, len(ordered))]]
    historical_signatures = {entry["signature"] for entry in ordered}
    candidates: list[dict[str, Any]] = [_base_candidate(best)]
    seen = {_candidate_signature(candidates[0])}

    if len(ordered) > 1:
        second = _base_candidate(ordered[1]["config"])
        second_sig = _candidate_signature(second)
        if second_sig not in seen:
            candidates.append(second)
            seen.add(second_sig)

    attempts = 0
    while len(candidates) < 8 and attempts < 5_000:
        attempts += 1
        parent_config = rng.choice(parents)
        candidate = _mutate_candidate(
            _base_candidate(parent_config),
            rng,
            intensity=1 if len(candidates) < 4 else 2,
        )
        signature = _candidate_signature(candidate)
        if signature in seen:
            continue
        if signature in historical_signatures:
            continue
        seen.add(signature)
        candidates.append(candidate)

    if len(candidates) < 8:
        for entry in ordered:
            candidate = _base_candidate(entry["config"])
            signature = _candidate_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(candidate)
            if len(candidates) == 8:
                break

    if len(candidates) < 8:
        raise RuntimeError("Unable to plan a full round of unique candidates")
    return candidates[:8]


def _next_run_dirs(runs_root: Path, count: int) -> list[Path]:
    existing = [int(path.name) for path in _iter_run_dirs(runs_root)]
    start = max(existing, default=0) + 1
    return [runs_root / f"{start + offset:04d}" for offset in range(count)]


def _launch_round(
    repo_root: Path,
    config_path: str,
    gpus: list[str],
    run_dirs: list[Path],
    candidates: list[dict[str, Any]],
    *,
    seed: int,
    round_index: int,
) -> list[dict[str, Any]]:
    if len(gpus) < len(candidates):
        raise ValueError(f"Need at least {len(candidates)} GPUs, got {len(gpus)}")
    launched = []
    for slot, (gpu, run_dir, candidate) in enumerate(zip(gpus, run_dirs, candidates, strict=True)):
        overrides = _candidate_to_overrides(candidate, round_index, slot, seed)
        command = [
            "uv",
            "run",
            "python",
            str(repo_root / "scripts" / "train.py"),
            "--config",
            config_path,
            "--run-dir",
            str(run_dir),
        ]
        for override in overrides:
            command.extend(["--set", override])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        launched.append(
            {
                "gpu": gpu,
                "run_dir": run_dir,
                "candidate": candidate,
                "command": command,
                "process": process,
            }
        )
    return launched


def _wait_round(launched: list[dict[str, Any]]) -> None:
    for record in launched:
        exit_code = record["process"].wait()
        if exit_code != 0:
            raise RuntimeError(f"Run failed: gpu={record['gpu']} run={record['run_dir']} exit_code={exit_code}")


def _print_round_summary(round_index: int, summaries: list[dict[str, Any]]) -> None:
    print(f"round={round_index} results")
    ranked = sorted(summaries, key=_rank_key)
    for summary in ranked[:8]:
        if summary["success"]:
            print(
                "  run=%s success time=%.3fs update=%s cov=%.6f sig=%s"
                % (
                    summary["run_dir"],
                    summary["success_elapsed_seconds"],
                    summary["success_update"],
                    summary["success_coverage"],
                    summary["signature"],
                )
            )
        else:
            print(
                "  run=%s miss best_cov=%.6f elapsed=%.3fs sig=%s"
                % (
                    summary["run_dir"],
                    summary["best_eval_coverage"],
                    summary["final_elapsed_seconds"],
                    summary["signature"],
                )
            )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = repo_root / args.runs_root
    runs_root.mkdir(parents=True, exist_ok=True)
    state_dir = repo_root / args.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    summary_path = state_dir / "rounds.jsonl"

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if len(gpus) != 8:
        raise ValueError(f"Expected exactly 8 GPUs for this search, got {gpus}")

    history = _load_history(runs_root)
    if not history:
        raise RuntimeError("No implementation4 history found to seed the search")

    search_rng = random.Random(args.seed + 99_000)
    for round_index in range(1, int(args.rounds) + 1):
        history = _load_history(runs_root)
        history.sort(key=_rank_key)
        best = history[0]
        print(
            "planning round=%s best_run=%s best_time=%s best_cov=%.6f"
            % (
                round_index,
                best["run_dir"],
                "nan" if best["success_elapsed_seconds"] is None else f"{best['success_elapsed_seconds']:.3f}s",
                best["best_eval_coverage"],
            )
        )
        candidates = _plan_round(history, round_index, search_rng)
        for slot, candidate in enumerate(candidates):
            print(f"  arm={slot} candidate={candidate}")

        run_dirs = _next_run_dirs(runs_root, len(candidates))
        launched = _launch_round(
            repo_root=repo_root,
            config_path=args.config,
            gpus=gpus,
            run_dirs=run_dirs,
            candidates=candidates,
            seed=args.seed,
            round_index=round_index,
        )
        round_start = time.perf_counter()
        _wait_round(launched)
        round_elapsed = time.perf_counter() - round_start

        round_summaries = []
        for record in launched:
            summary = _summarize_run(record["run_dir"])
            if summary is None:
                raise RuntimeError(f"Missing summary for run {record['run_dir']}")
            round_summaries.append(summary)
        _print_round_summary(round_index, round_summaries)

        history = _load_history(runs_root)
        best = history[0]
        round_record = {
            "round": round_index,
            "round_elapsed_seconds": round_elapsed,
            "best_run": best["run_dir"],
            "best_success_elapsed_seconds": best["success_elapsed_seconds"],
            "best_success_update": best["success_update"],
            "best_eval_coverage": best["best_eval_coverage"],
            "launched_runs": [
                {
                    "run_dir": record["run_dir"].name,
                    "gpu": record["gpu"],
                    "candidate": record["candidate"],
                }
                for record in launched
            ],
            "round_results": [
                {
                    "run_dir": summary["run_dir"],
                    "success": summary["success"],
                    "success_elapsed_seconds": summary["success_elapsed_seconds"],
                    "success_update": summary["success_update"],
                    "best_eval_coverage": summary["best_eval_coverage"],
                }
                for summary in round_summaries
            ],
        }
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(round_record) + "\n")
        print(
            "completed round=%s round_elapsed=%.3fs overall_best_run=%s overall_best_time=%s"
            % (
                round_index,
                round_elapsed,
                best["run_dir"],
                "nan" if best["success_elapsed_seconds"] is None else f"{best['success_elapsed_seconds']:.3f}s",
            )
        )


if __name__ == "__main__":
    main()
