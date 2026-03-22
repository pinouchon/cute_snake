from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    success: bool
    success_update: int | None
    success_elapsed_seconds: float | None
    final_elapsed_seconds: float
    final_eval_coverage: float | None
    final_eval_return: float | None
    best_eval_coverage: float | None
    env_steps: int
    config: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--selection-target", type=float, default=0.80)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pattern", default="[0-9][0-9][0-9][0-9]")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping config at {path}, got {type(data)!r}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping json at {path}, got {type(data)!r}")
    return data


def _summarize_run(run_dir: Path, selection_target: float) -> RunSummary | None:
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    if not config_path.exists() or not metrics_path.exists():
        return None

    config = _load_yaml(config_path)
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = [json.loads(line) for line in handle if line.strip()]
    if not metrics:
        return None

    final = metrics[-1]
    best_eval = None
    for record in metrics:
        eval_cov = record.get("eval_mean_final_coverage")
        if eval_cov is None:
            continue
        best_eval = eval_cov if best_eval is None else max(best_eval, float(eval_cov))

    success_record = None
    for record in metrics:
        eval_cov = record.get("eval_mean_final_coverage")
        if eval_cov is not None and float(eval_cov) >= selection_target:
            success_record = record
            break

    final_eval_coverage = final.get("eval_mean_final_coverage")
    final_eval_return = final.get("eval_mean_episode_return")
    if final_eval_coverage is None:
        eval_path = run_dir / "eval.json"
        if eval_path.exists():
            eval_data = _load_json(eval_path)
            final_eval_coverage = eval_data.get("mean_final_coverage")
            final_eval_return = eval_data.get("mean_episode_return")

    return RunSummary(
        run_dir=run_dir,
        success=success_record is not None,
        success_update=int(success_record["update"]) if success_record else None,
        success_elapsed_seconds=float(success_record["elapsed_seconds"]) if success_record else None,
        final_elapsed_seconds=float(final["elapsed_seconds"]),
        final_eval_coverage=float(final_eval_coverage) if final_eval_coverage is not None else None,
        final_eval_return=float(final_eval_return) if final_eval_return is not None else None,
        best_eval_coverage=float(best_eval) if best_eval is not None else None,
        env_steps=int(final["env_steps"]),
        config=config,
    )


def _config_brief(config: dict[str, Any]) -> str:
    keys = [
        "num_envs",
        "rollout_steps",
        "minibatches",
        "update_epochs",
        "learning_rate",
        "amp",
        "use_cute_step_core",
    ]
    parts = []
    for key in keys:
        if key in config:
            parts.append(f"{key}={config[key]}")
    return ", ".join(parts)


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.6f}"


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = repo_root / args.runs_root
    summaries: list[RunSummary] = []
    for run_dir in sorted(runs_root.glob(args.pattern)):
        if run_dir.is_dir():
            summary = _summarize_run(run_dir, args.selection_target)
            if summary is not None:
                summaries.append(summary)

    if not summaries:
        print("no completed runs found")
        return

    successful = sorted(
        [summary for summary in summaries if summary.success],
        key=lambda summary: (
            summary.success_elapsed_seconds if summary.success_elapsed_seconds is not None else float("inf"),
            -(summary.final_eval_coverage or -1.0),
            summary.run_dir.name,
        ),
    )
    fallback = sorted(
        summaries,
        key=lambda summary: (
            -(summary.best_eval_coverage or -1.0),
            summary.final_elapsed_seconds,
            summary.run_dir.name,
        ),
    )

    print("Successful runs:")
    if successful:
        for summary in successful[: args.top_k]:
            print(
                f"{summary.run_dir.name} cov={summary.final_eval_coverage:.6f} "
                f"time_to_target={summary.success_elapsed_seconds:.3f}s "
                f"update={summary.success_update} "
                f"env_steps={summary.env_steps} "
                f"{_config_brief(summary.config)}"
            )
    else:
        print("none")

    print("Best runs by coverage:")
    for summary in fallback[: args.top_k]:
        print(
            f"{summary.run_dir.name} best_cov={_fmt_float(summary.best_eval_coverage)} "
            f"final_cov={_fmt_float(summary.final_eval_coverage)} "
            f"elapsed={summary.final_elapsed_seconds:.3f}s "
            f"{_config_brief(summary.config)}"
        )


if __name__ == "__main__":
    main()
