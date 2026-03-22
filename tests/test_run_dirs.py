from __future__ import annotations

from snake.run_dirs import allocate_run_dir


def test_allocate_run_dir_increments(tmp_path) -> None:
    first = allocate_run_dir(tmp_path)
    second = allocate_run_dir(tmp_path)
    assert first.name == "0001"
    assert second.name == "0002"
    assert (first / "checkpoints").is_dir()
