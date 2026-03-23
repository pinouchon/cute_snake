from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import optuna
import torch

from snake.config import apply_overrides, load_yaml_config, normalize_config, save_yaml_config
from snake.implementations.implementation4 import train
from snake.run_dirs import allocate_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/implementation4.yaml")
    parser.add_argument("--search-config", default="configs/optuna_implementation4.yaml")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--search-set", dest="search_overrides", action="append", default=[])
    return parser.parse_args()


def _suggest_value(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    kind = str(spec["type"])
    if kind == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
            step=spec.get("step"),
        )
    if kind == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if kind == "bool":
        return trial.suggest_categorical(name, [False, True])
    raise ValueError(f"Unsupported search-space type {kind!r} for {name}")


def _prepare_storage(storage: str | None) -> None:
    if storage is None or not storage.startswith("sqlite:///"):
        return
    db_path = Path(storage.removeprefix("sqlite:///"))
    db_path.parent.mkdir(parents=True, exist_ok=True)


def _summarize_run(metrics_path: Path, success_target: float) -> dict[str, float | bool]:
    first_success_elapsed: float | None = None
    best_coverage = float("-inf")
    last_elapsed = 0.0
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        last_elapsed = float(record.get("elapsed_seconds", last_elapsed))
        eval_coverage = record.get("eval_mean_final_coverage")
        if eval_coverage is None:
            continue
        eval_coverage = float(eval_coverage)
        best_coverage = max(best_coverage, eval_coverage)
        if first_success_elapsed is None and eval_coverage >= success_target:
            first_success_elapsed = float(record["elapsed_seconds"])
    return {
        "success": first_success_elapsed is not None,
        "elapsed_seconds": first_success_elapsed if first_success_elapsed is not None else last_elapsed,
        "best_eval_coverage": best_coverage if best_coverage != float("-inf") else 0.0,
    }


def _build_sampler(study_config: dict[str, Any], worker_id: int = 0) -> optuna.samplers.BaseSampler:
    seed = int(study_config.get("seed", 92)) + worker_id
    sampler_name = str(study_config.get("sampler", "tpe")).lower()
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    sampler_options = dict(study_config.get("sampler_options", {}))
    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=int(sampler_options.get("n_startup_trials", 8)),
        n_ei_candidates=int(sampler_options.get("n_ei_candidates", 48)),
        multivariate=bool(sampler_options.get("multivariate", True)),
        group=bool(sampler_options.get("group", False)),
        constant_liar=bool(sampler_options.get("constant_liar", True)),
    )


def _visible_gpus() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [str(index) for index in range(torch.cuda.device_count())]


def _selected_gpus(raw: str) -> list[str]:
    if raw == "all":
        return _visible_gpus()
    return [part.strip() for part in raw.split(",") if part.strip()]


def _trial_splits(total_trials: int, workers: int) -> list[int]:
    base = total_trials // workers
    remainder = total_trials % workers
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def _launch_parallel_workers(args: argparse.Namespace) -> bool:
    if args.worker:
        return False
    gpu_ids = _selected_gpus(args.gpus)
    if len(gpu_ids) <= 1:
        return False
    worker_count = args.workers if args.workers > 0 else len(gpu_ids)
    worker_count = max(1, min(worker_count, len(gpu_ids), args.n_trials))
    if worker_count <= 1:
        return False

    trial_splits = _trial_splits(args.n_trials, worker_count)
    processes: list[subprocess.Popen[str]] = []
    script_path = Path(__file__).resolve()
    for worker_index, trial_count in enumerate(trial_splits):
        if trial_count <= 0:
            continue
        command = [
            sys.executable,
            str(script_path),
            "--worker",
            "--worker-id",
            str(worker_index),
            "--config",
            args.config,
            "--search-config",
            args.search_config,
            "--n-trials",
            str(trial_count),
        ]
        if args.study_name is not None:
            command.extend(["--study-name", args.study_name])
        if args.storage is not None:
            command.extend(["--storage", args.storage])
        if args.timeout is not None:
            command.extend(["--timeout", str(args.timeout)])
        for override in args.overrides:
            command.extend(["--set", override])
        for override in args.search_overrides:
            command.extend(["--search-set", override])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_index]
        env["PYTHONUNBUFFERED"] = "1"
        processes.append(subprocess.Popen(command, env=env, text=True))

    exit_code = 0
    for process in processes:
        return_code = process.wait()
        if return_code != 0 and exit_code == 0:
            exit_code = return_code
    if exit_code != 0:
        raise SystemExit(exit_code)
    return True


def main() -> None:
    args = parse_args()
    base_config = normalize_config(apply_overrides(load_yaml_config(args.config), args.overrides))
    search_config = apply_overrides(load_yaml_config(args.search_config), args.search_overrides)
    study_config = dict(search_config.get("study", {}))
    sweep_config = dict(search_config.get("sweep", {}))
    fixed_overrides = dict(sweep_config.get("fixed_overrides", {}))
    search_space = dict(sweep_config.get("search_space", {}))
    if args.gpus is None:
        args.gpus = str(sweep_config.get("gpus", "all"))
    if args.workers is None:
        args.workers = int(sweep_config.get("workers", 0))

    study_name = args.study_name or str(study_config.get("study_name", "implementation4"))
    storage = args.storage or study_config.get("storage")
    _prepare_storage(storage)

    if not args.worker and len(_selected_gpus(args.gpus)) > 1 and max(1, args.n_trials) > 1:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=str(study_config.get("direction", "minimize")),
            load_if_exists=True,
            sampler=_build_sampler(study_config),
        )
    if _launch_parallel_workers(args):
        study = optuna.load_study(study_name=study_name, storage=storage)
        best = study.best_trial
        print(
            json.dumps(
                {
                    "study_name": study.study_name,
                    "best_trial": best.number,
                    "best_value": best.value,
                    "best_params": best.params,
                    "best_run_dir": best.user_attrs.get("run_dir"),
                },
                indent=2,
            )
        )
        return

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=str(study_config.get("direction", "minimize")),
        load_if_exists=bool(study_config.get("load_if_exists", True)),
        sampler=_build_sampler(study_config, worker_id=args.worker_id),
    )

    failure_score = float(study_config.get("failure_score", 1000.0))
    failure_gap_scale = float(study_config.get("failure_gap_scale", 100.0))

    def objective(trial: optuna.Trial) -> float:
        config = deepcopy(base_config)
        config.update(fixed_overrides)
        for name, spec in search_space.items():
            config[name] = _suggest_value(trial, name, spec)
        if args.overrides:
            config = apply_overrides(config, args.overrides)

        run_dir = allocate_run_dir(config.get("run_root", "runs"))
        save_yaml_config(config, run_dir / "config.yaml")
        trial.set_user_attr("run_dir", str(run_dir))

        logger = logging.getLogger(f"snake.optuna.{trial.number}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())
        logger.addHandler(logging.FileHandler(run_dir / "train.log", encoding="utf-8"))

        score = failure_score
        try:
            result = train(config, run_dir, logger)
            summary = _summarize_run(run_dir / str(config["metrics_file"]), float(config["success_target"]))
            trial.set_user_attr("success", bool(summary["success"]))
            trial.set_user_attr("best_eval_coverage", float(summary["best_eval_coverage"]))
            trial.set_user_attr("elapsed_seconds", float(summary["elapsed_seconds"]))
            trial.set_user_attr("best_eval_mean_final_coverage", float(result["best_eval_mean_final_coverage"]))
            if bool(summary["success"]):
                score = float(summary["elapsed_seconds"])
            else:
                score = (
                    failure_score
                    + failure_gap_scale * max(0.0, float(config["success_target"]) - float(summary["best_eval_coverage"]))
                    + float(summary["elapsed_seconds"])
                )
        except Exception as exc:
            trial.set_user_attr("success", False)
            trial.set_user_attr("error", repr(exc))
            logger.exception("trial_failed")
            score = failure_score * 10.0

        with (run_dir / "optuna_trial.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "trial_number": trial.number,
                    "score": score,
                    "params": trial.params,
                    "study_name": study_name,
                },
                handle,
                indent=2,
            )
        return score

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    if not args.worker:
        best = study.best_trial
        print(
            json.dumps(
                {
                    "study_name": study.study_name,
                    "best_trial": best.number,
                    "best_value": best.value,
                    "best_params": best.params,
                    "best_run_dir": best.user_attrs.get("run_dir"),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
