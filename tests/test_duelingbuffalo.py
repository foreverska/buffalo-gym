from collections import deque

import numpy as np
import gymnasium as gym
import buffalo_gym.envs.buffalo_gym

def test_duelingbuffalo():
    env = gym.make('DuelingBuffalo-v0', arms=2)

    obs, info = env.reset()

    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert obs[0] in (0, 1)

    high_arm = np.argmax(info['offsets'])
    low_arm = np.argmin(info['offsets'])

    _, reward, *_ = env.step([high_arm, low_arm])
    assert reward == 1

    _, reward, *_ = env.step([low_arm, high_arm])
    assert reward == 0

def test_duelingbuffalo_edges():
    env = gym.make('DuelingBuffalo-v0', arms=2)

    _, _ = env.reset()

    try:
        env.step(np.array([1, 1]))
        assert False
    except ValueError:
        assert True

    try:
        env.step(np.array([0]))
        assert False
    except ValueError:
        assert True