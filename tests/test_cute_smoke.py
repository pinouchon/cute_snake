from __future__ import annotations

import pytest
import torch

from snake.cute_kernels import HAVE_CUTE, run_cute_smoke


@pytest.mark.skipif(not HAVE_CUTE, reason="CuTe DSL is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cute_smoke_kernel_runs() -> None:
    result = run_cute_smoke()
    expected = torch.arange(1, 17, dtype=torch.float32, device="cuda")
    assert torch.equal(result, expected)
