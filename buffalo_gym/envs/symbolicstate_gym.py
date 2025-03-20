from typing import Any, TypeVar, SupportsFloat
import random

import gymnasium as gym
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class SymbolicStateEnv(gym.Env):
    """
    Multi-armed bandit environment with multiple states where the states do not correlate with rewards
    """
    metadata = {'render_modes': []}

    def __draw_arms(self):
        """
        Draw new arms
        """
        self.rng = np.random.default_rng(self.seed)
        self.offsets = np.random.uniform(self.min_suboptimal_mean, self.max_suboptimal_mean,
                                         size=self.arms)

        optimal_arms = self.rng.choice(range(self.arms), self.optimal_arms, replace=False)
        for arm in optimal_arms:
            self.offsets[arm] = self.optimal_mean
        self.stds = [self.optimal_std if arm in optimal_arms else self.suboptimal_std for arm in range(self.arms)]

    def __init__(self, arms: int = 10, states: int = 2, optimal_arms: int | list[int] = 1,
                 dynamic_rate: int | None = None, pace: int = 5, seed: int | None = None, optimal_mean: float = 10,
                 optimal_std: float = 1, min_suboptimal_mean: float = 0, max_suboptimal_mean: float = 5,
                 suboptimal_std: float = 1):
        """
        Multi-armed bandit environment with k arms and n states, states are uninformative
        :param arms: number of arms
        :param states: number of states
        :param optimal_arms: number of optimal arms or list of optimal orms in each state
        :param dynamic_rate: number of steps between drawing new arm means, None means no dynamic rate
        :param pace: number of steps between drawing new state
        :param seed: random seed
        :param optimal_mean: mean of optimal arms
        :param optimal_std: std of optimal arms
        :param min_suboptimal_mean: min mean of suboptimal arms
        :param max_suboptimal_mean: max mean of suboptimal arms
        :param suboptimal_std: std of suboptimal arms
        """
        self.arms = arms
        self.states = states
        self.dynamic_rate = dynamic_rate
        self.pace = pace
        self.initial_seed = seed
        self.seed = seed
        self.optimal_mean = optimal_mean
        self.optimal_std = optimal_std
        self.min_suboptimal_mean = min_suboptimal_mean
        self.max_suboptimal_mean = max_suboptimal_mean
        self.suboptimal_std = suboptimal_std

        if optimal_arms is list and len(optimal_arms) != self.arms:
            raise ValueError("Optimal arms list must have equal number of arms")
        self.optimal_arms = optimal_arms

        self.action_space = gym.spaces.Discrete(arms)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.pulls = 0
        self.ssr = 0
        self.state = 0

        self.__draw_arms()

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

        self.seed = seed
        self.pulls = 0
        self.ssr = 0
        self.state = 0
        self.__draw_arms()

        return np.zeros((1,), dtype=np.float32), {"offsets": self.offsets}

    def step(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Steps the environment
        :param action: arm to pull
        :return: observation, reward, done, term, info
        """
        reward = self.rng.normal(self.offsets[action], self.stds[action], 1)[0]

        self.ssr += 1
        if self.pace is None or self.ssr % self.pace == 0:
            self.state = np.random.randint(0, self.states)

        self.pulls += 1
        if self.dynamic_rate is not None and self.pulls % self.dynamic_rate == 0:
            if self.seed is not None:
                self.seed += 1
            self.__draw_arms()

        return np.ones((1,), dtype=np.float32)*self.state, reward, False, False, {'offsets': self.offsets}
