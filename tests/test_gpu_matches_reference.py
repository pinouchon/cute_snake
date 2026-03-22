from __future__ import annotations

import pytest
import torch

from snake.env_gpu import TorchSnakeBatchEnv
from snake.env_reference import ReferenceSnakeEnv


def test_single_env_matches_reference_for_fixed_actions() -> None:
    seed = 17
    actions = [1, 1, 2, 2, 3, 0, 0]

    ref = ReferenceSnakeEnv(seed=seed)
    ref.reset(seed=seed)

    env = TorchSnakeBatchEnv(num_envs=1, seed=seed, device="cpu")
    env.reset()

    for action in actions:
        _, ref_reward, ref_done, ref_info = ref.step(action)
        _, rewards, dones, info = env.step(torch.tensor([action]))

        assert rewards[0].item() == pytest.approx(ref_reward)
        assert bool(dones[0].item()) == ref_done
        assert env.snapshot(0)["snake"] == ref.snapshot()["snake"]
        assert env.snapshot(0)["heading"] == ref.snapshot()["heading"]
        assert env.snapshot(0)["food"] == ref.snapshot()["food"]
        assert info["final_coverage"][0].item() == pytest.approx(ref_info["final_coverage"])

        if ref_done:
            break
