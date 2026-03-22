set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default: train

setup:
    uv sync --extra dev

test:
    uv run pytest

next-run-dir:
    @mkdir -p runs
    @last="$$(find runs -maxdepth 1 -mindepth 1 -type d -regex '.*/[0-9][0-9][0-9][0-9]' | sed 's#.*/##' | sort | tail -n 1)"; \
    if [ -z "$$last" ]; then next=1; else next=$$((10#$$last + 1)); fi; \
    run_dir=$$(printf 'runs/%04d' "$$next"); \
    mkdir -p "$$run_dir/checkpoints" "$$run_dir/visualizations" "$$run_dir/profiler"; \
    printf '%s\n' "$$run_dir"

train RUN_DIR='':
    @run_dir="$${RUN_DIR:-$$(just --quiet next-run-dir)}"; \
    case "$$run_dir" in \
      runs/*) ;; \
      [0-9][0-9][0-9][0-9]) run_dir="runs/$$run_dir" ;; \
    esac; \
    uv run python scripts/train.py --config configs/implementation4.yaml --run-dir "$$run_dir"

eval RUN_DIR='':
    @run_dir="$${RUN_DIR:-$$(find runs -maxdepth 1 -mindepth 1 -type d -regex '.*/[0-9][0-9][0-9][0-9]' | sort | tail -n 1)}"; \
    test -n "$$run_dir"; \
    case "$$run_dir" in \
      runs/*) ;; \
      [0-9][0-9][0-9][0-9]) run_dir="runs/$$run_dir" ;; \
    esac; \
    uv run python scripts/eval.py --run-dir "$$run_dir"

visualize RUN_DIR='':
    @run_dir="$${RUN_DIR:-$$(find runs -maxdepth 1 -mindepth 1 -type d -regex '.*/[0-9][0-9][0-9][0-9]' | sort | tail -n 1)}"; \
    test -n "$$run_dir"; \
    case "$$run_dir" in \
      runs/*) ;; \
      [0-9][0-9][0-9][0-9]) run_dir="runs/$$run_dir" ;; \
    esac; \
    uv run python scripts/visualize.py --run-dir "$$run_dir"
