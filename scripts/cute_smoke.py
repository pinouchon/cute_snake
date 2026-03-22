from __future__ import annotations

import torch

from snake.cute_kernels import HAVE_CUTE, run_cute_smoke


def main() -> None:
    if not HAVE_CUTE:
        raise SystemExit("CuTe DSL is not available in this environment")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the CuTe smoke test")

    result = run_cute_smoke()
    expected = torch.arange(1, 17, dtype=torch.float32, device="cuda")
    if not torch.equal(result, expected):
        raise SystemExit(f"unexpected result: {result}")
    print("cute_smoke_ok", result.tolist())


if __name__ == "__main__":
    main()
