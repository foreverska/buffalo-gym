
import numpy as np
import gymnasium as gym
import buffalo_gym.envs.buffalo_gym

def test_buffalo():
    env = gym.make('Buffalo-v0')

    obs, info = env.reset()

    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert obs[0] == 0

    obs, reward, done, term, info = env.step(env.action_space.sample())

    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert obs[0] == 0
    assert done is False
    assert term is False

    assert 1
