from typing import Any, TypeVar, SupportsFloat
import random

import gymnasium as gym
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class MultiBuffaloEnv(gym.Env):
    """
    Multi-armed bandit environment with multiple states that have (probably) distinct rewards
    """
    metadata = {'render_modes': []}

    def __init__(self, arms: int = 10, states: int = 2, pace: int | None = None):
        """
        Multi-armed bandit environment with k arms and n states
        :param arms: number of arms
        :param states: number of states
        :param pace: number of steps between state changes, None for every step
        """
        self.action_space = gym.spaces.Discrete(arms)
        self.observation_space = gym.spaces.Box(low=0, high=states, shape=(1,), dtype=np.float32)

        self.offsets = np.random.normal(0, arms, (arms, states))
        self.pace = pace
        self.states = states
        self.state = 0
        self.ssr = 0

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets the environment
        :param seed: WARN unused, defaults to None
        :param options: WARN unused, defaults to None
        :return: observation, info
        """

        self.state = 0
        self.ssr = 0

        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Steps the environment
        :param action: arm to pull
        :return: observation, reward, done, term, info
        """
        reward = np.random.normal(0, 1, 1)[0] + self.offsets[action, self.state]
        self.ssr += 1
        if self.pace is None or self.ssr % self.pace == 0:
            self.state = random.randint(0, self.states - 1)

        return np.ones((1,), dtype=np.float32)*self.state, reward, False, False, {}
