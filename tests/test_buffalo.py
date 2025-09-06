
import numpy as np
import gymnasium as gym
import buffalo_gym.envs.buffalo_gym

def test_buffalo():
    env = gym.make('Buffalo-v0')

    obs, info = env.reset(seed=42)

    assert np.allclose(info['offsets'][0], 0.0, rtol=1e-3)

    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert obs[0] == 0

    obs, reward, done, term, info = env.step(env.action_space.sample())

    assert np.allclose(info['offsets'][0], [10.0, 4.292, 3.486, 0.471, 4.878, 3.806, 3.93, 0.641, 2.252, 1.854],
                       rtol=1e-3)

    assert obs.shape == (1,)
    assert obs.dtype == np.float32
    assert obs[0] == 0
    assert done is False
    assert term is False

    assert 1

def test_buffalo_acceleration():
    env = gym.make('Buffalo-v0', min_suboptimal_mean=-1, arm_acceleration=2.0)

    _, info = env.reset(seed=42)

    assert np.allclose(info['offsets'][0], 0.0, rtol=1e-3)

    _, _, _, _, info = env.step(env.action_space.sample())

    assert np.allclose(info['offsets'][0], [ 2.,  2.,  2., -0.4349,  2., 2.,  2., -0.2313 ,  1.7023,  1.2247], rtol=1e-3)




