
import numpy as np
import gymnasium as gym
import buffalo_gym.envs.buffalo_gym

def test_buffalotrail():
    env = gym.make('BuffaloTrail-v0')

    obs, info = env.reset()

    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert obs[0] == 0

    last_action = env.action_space.sample()
    obs, reward, done, term, info = env.step(last_action)

    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert obs[0] == last_action
    assert done is False
    assert term is False

    assert 1

def test_buffalotrail_max():
    env = gym.make('BuffaloTrail-v0')

    obs, info = env.reset()

    sequence = info['goal']

    last_reward = 0
    last_action = 0
    for action in sequence:
        obs, reward, done, term, info = env.step(action)
        last_action = action
        last_reward = reward

    assert last_reward == 100
    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert obs[0] == last_action
    assert done is False
    assert term is False

    assert 1
