from typing import Any, TypeVar, SupportsFloat

import gymnasium as gym
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class BoundlessBuffaloEnv(gym.Env):

    def __draw_polynomial(self):
        """
        Draw a new set of coefficients for the reward polynomial
        """
        self.rng = np.random.default_rng(self.seed)
        self.coefficients = [self.rng.uniform(-0.1, 0.1) for x in range(self.degree)]

    def __init__(self, degree: int = 3, dynamic_rate: int | None = None, seed: int | None = None,
                 std_deviation: float = 0.1):
        """
        Infinite armed bandit environment.  The input is scaled from (-inf, +inf) to (-1, +1) in an attempt to keep
        this numerically stable.  Also, coefficients are drawn from (-0.1, 0.1) to help this along.
        :param degree: Degree of polynomial which defines the reward function
        :param dynamic_rate: number of pulls between drawing a new polynomial, NONE if not dynamic
        :param seed: Randomness seed, NONE if it doesn't matter
        :param std_deviation: randomness around reward function
        """

        self.initial_seed = seed
        self.seed = seed
        self.degree = degree
        self.dynamic_rate = dynamic_rate
        self.std_deviation = std_deviation

        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.__draw_polynomial()
        self.pulls = 0

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        """"
        Resets the environment
        :param seed: WARN unused, defaults to None
        :param options: WARN unused, defaults to None
        :return: observation, info
        """
        self.seed = self.initial_seed
        self.__draw_polynomial()
        self.pulls = 0

        return np.zeros((1,), dtype=np.float32), {"coef": self.coefficients}

    def step(self, action: float) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Steps the environment
        :param action: One of infinite arms to pull in (-inf, +inf)
        :return: observation, reward, done, term, info
        """
        arm = (action / np.sqrt(1 + np.power(action, 2)))[0]
        reward = sum([c * np.power(arm, n) for n, c in enumerate(self.coefficients)])
        reward += self.rng.normal(self.std_deviation)

        self.pulls += 1
        if self.dynamic_rate is not None and self.pulls % self.dynamic_rate == 0:
            if self.seed is not None:
                self.seed += 1
            self.__draw_polynomial()

        return np.zeros((1,), dtype=np.float32), reward, False, False, {"coef": self.coefficients}
