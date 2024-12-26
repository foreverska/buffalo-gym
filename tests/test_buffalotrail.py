from collections import deque

import numpy as np
import gymnasium as gym
import buffalo_gym.envs.buffalo_gym

def test_multibuffalo():
    env = gym.make('BuffaloTrail-v0')

    obs, info = env.reset()

    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert obs[0] in (0, 1)

    states = []
    for _ in range(20):
        obs, reward, done, term, info = env.step(env.action_space.sample())

        assert obs.shape == (1,)
        assert obs.dtype == np.float32
        assert obs[0] in (0, 1)
        states.append(obs[0])
        assert done is False
        assert term is False

    assert set(states) == {0, 1}

def test_multibuffalo_threestates():
    env = gym.make('BuffaloTrail-v0', states=3, pace=2)

    obs, info = env.reset()

    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert obs[0] in (0, 1, 2)

    states = set()
    for _ in range(100):
        obs, reward, done, term, info = env.step(env.action_space.sample())

        assert obs.shape == (1,)
        assert obs.dtype == np.float32
        assert obs[0] in (0, 1, 2)
        states.add(obs[0])
        assert done is False
        assert term is False

        if states == {0.0, 1.0, 2.0}:
            break

    assert states == {0.0, 1.0, 2.0}

def test_multibuffalo_trail():
    env = gym.make('BuffaloTrail-v0', states=3, pace=1, sequence_length=2, goal_reward=100)

    obs, info = env.reset()

    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert obs[0] in (0, 1, 2)

    trail = info['goal']
    trail_action = info['goal_action']

    sequence = deque([], maxlen=2)
    while True:
        obs, reward, done, term, info = env.step(env.action_space.sample())
        sequence.append(int(obs[0]))
        if sum(1 if a == b else 0 for (a, b) in zip(trail, sequence)) == 2:
            break

    _, reward, _, _, _ = env.step(trail_action)
    assert float(reward) > 90.0
