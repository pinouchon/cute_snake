from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def allocate_run_dir(root: str | Path = "runs", explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        run_dir = Path(explicit)
        if run_dir.exists():
            if any(run_dir.iterdir()):
                raise FileExistsError(f"Explicit run directory already exists and is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        while True:
            existing = [
                int(path.name)
                for path in root_path.iterdir()
                if path.is_dir() and path.name.isdigit()
            ]
            run_dir = root_path / f"{(max(existing, default=0) + 1):04d}"
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                continue
    for child in ("checkpoints", "visualizations", "profiler"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
