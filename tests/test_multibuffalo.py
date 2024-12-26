
import numpy as np
import gymnasium as gym
import buffalo_gym

def test_multibuffalo():
    env = gym.make('MultiBuffalo-v0')

    obs, info = env.reset()

    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert obs[0] in (0, 1)

    states = set()
    for _ in range(100):
        obs, reward, done, term, info = env.step(env.action_space.sample())

        assert obs.shape == (1,)
        assert obs.dtype == np.float32
        assert obs[0] in (0, 1)
        states.add(obs[0])
        assert done is False
        assert term is False

        if states == {0.0, 1.0}:
            break

    assert states == {0.0, 1.0}

def test_multibuffalo_threestates():
    env = gym.make('MultiBuffalo-v0', states=3, pace=2)

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
