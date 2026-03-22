from __future__ import annotations

import pytest

from snake.env_reference import ReferenceSnakeEnv


def test_invalid_reverse_continues_straight() -> None:
    env = ReferenceSnakeEnv()
    env.reset(seed=123)
    _, _, _, _ = env.step(3)
    snapshot = env.snapshot()
    assert snapshot["heading"] == 1
    assert snapshot["snake"][-1] == 4 * 8 + 4


def test_eating_increases_length_and_reward() -> None:
    env = ReferenceSnakeEnv()
    env.reset(seed=123)
    env.force_state(snake=[33, 34, 35], heading=1, food=36)
    _, reward, done, info = env.step(1)
    snapshot = env.snapshot()
    assert not done
    assert reward == pytest.approx(0.99)
    assert len(snapshot["snake"]) == 4
    assert info["final_coverage"] == pytest.approx(4 / 64)


def test_wall_collision_ends_episode() -> None:
    env = ReferenceSnakeEnv()
    env.reset(seed=123)
    env.force_state(snake=[5, 6, 7], heading=1, food=10)
    _, reward, done, info = env.step(1)
    assert done
    assert reward == pytest.approx(-1.01)
    assert info["episode_length"] == 1
