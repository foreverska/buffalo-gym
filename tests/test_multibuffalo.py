
import numpy as np
import gymnasium as gym
import buffalo_gym

def test_multibuffalo():
    env = gym.make('MultiBuffalo-v0')

    obs, info = env.reset()

    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert obs[0] in (0, 1)

    states = []
    for _ in range(10):
        obs, reward, done, term, info = env.step(env.action_space.sample())

        assert obs.shape == (1,)
        assert obs.dtype == np.float32
        assert obs[0] in (0, 1)
        states.append(obs[0])
        assert done is False
        assert term is False

    assert set(states) == {0, 1}

def test_multibuffalo_threestates():
    env = gym.make('MultiBuffalo-v0', states=3)

    obs, info = env.reset()

    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert obs[0] in (0, 1, 2)

    states = []
    for _ in range(10):
        obs, reward, done, term, info = env.step(env.action_space.sample())

        assert obs.shape == (1,)
        assert obs.dtype == np.float32
        assert obs[0] in (0, 1, 2)
        states.append(obs[0])
        assert done is False
        assert term is False

    assert set(states) == {0, 1, 2}
