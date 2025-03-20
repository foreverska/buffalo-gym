
import numpy as np
import gymnasium as gym
import buffalo_gym.envs.buffalo_gym

def test_buffalo():
    env = gym.make('BoundlessBuffalo-v0')

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

def test_high_degree():
    env = gym.make('BoundlessBuffalo-v0', degree=6)

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