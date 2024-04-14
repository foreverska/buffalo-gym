from typing import Any, TypeVar, SupportsFloat

import gymnasium as gym
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BuffaloEnv(gym.Env):
    """
    Standard multi-armed bandit environment with static reward distributions.
    """
    metadata = {'render_modes': []}

    def __init__(self, arms: int = 10):
        """
        Multi-armed bandit environment with k-static valued arms
        :param arms: number of arms
        """
        self.action_space = gym.spaces.Discrete(arms)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.offsets = np.random.normal(0, arms, (arms,))

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

        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Steps the environment
        :param action: arm to pull
        :return: observation, reward, done, term, info
        """
        reward = np.random.normal(0, 1, 1)[0] + self.offsets[action]

        return np.zeros((1,), dtype=np.float32), reward, False, False, {}
