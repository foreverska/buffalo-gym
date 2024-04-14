import gymnasium as gym
import buffalo_gym.envs.buffalo_gym

def test_buffalo():
    env = gym.make('Buffalo-v0')

    env.reset()
    env.step(env.action_space.sample())

    assert 1
