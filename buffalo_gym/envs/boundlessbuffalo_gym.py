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
        coefficients = [self.rng.uniform(-self.coef_range, self.coef_range) for _ in range(self.degree + 1)]
        coefficients[0] = 0
        if coefficients[-1] > 0:
            coefficients[-1] *= -1

        self.polynomial = np.polynomial.Polynomial(coefficients)
        d1: np.polynomial.Polynomial = self.polynomial.deriv()
        d2 = d1.deriv()

        roots = [root.real for root in d1.roots() if np.isrealobj(root) and d2(root.real) < 0]
        maximum = max([self.polynomial(root) for root in roots], default=0)
        self.polynomial.coef[0] += self.max_val - maximum

        self.left_shoulder = -np.inf
        self.right_shoulder = np.inf
        if self.shoulders:
            roots = [root.real for root in d1.roots() if np.isrealobj(root)]
            minimum = min([self.polynomial(root) for root in roots], default=0)
            minimum -= abs(minimum)
            cross = np.polynomial.Polynomial(self.polynomial.coef)
            cross.coef[0] += -minimum
            shoulders = [root.real for root in cross.roots()]
            self.left_shoulder = min(shoulders, default=-np.inf)
            self.right_shoulder = max(shoulders, default=np.inf)

    def __init__(self, degree: int = 2, dynamic_rate: int | None = None, seed: int | None = None,
                 std_deviation: float = 0.1, coef_range: float = 10, max_val: float = 10.0, shoulders: bool = True,
                 shoulder_leakage: float = 0.0):
        """
        Infinite armed bandit environment.  The input is scaled from (-inf, +inf) to (-1, +1) in an attempt to keep
        this numerically stable.  Also, coefficients are drawn from (-0.1, 0.1) to help this along.
        :param degree: Degree of polynomial which defines the reward function
        :param dynamic_rate: number of pulls between drawing a new polynomial, NONE if not dynamic
        :param seed: Randomness seed, NONE if it doesn't matter
        :param std_deviation: randomness around reward function
        """
        if degree < 2 or degree % 2 == 1:
            raise ValueError("degree must be an even number greater than or equal to 2")

        self.initial_seed = seed
        self.seed = seed
        self.degree = degree
        self.dynamic_rate = dynamic_rate
        self.std_deviation = std_deviation
        self.coef_range = coef_range
        self.max_val = max_val
        self.shoulders = shoulders
        self.shoulder_leakage = shoulder_leakage

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
        self.seed = seed
        self.__draw_polynomial()
        self.pulls = 0

        return np.zeros((1,), dtype=np.float32), {"coef": self.polynomial.coef}

    def step(self, action: float) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Steps the environment
        :param action: One of infinite arms to pull in (-inf, +inf)
        :return: observation, reward, done, term, info
        """
        if self.shoulders and action < self.left_shoulder:
            left = self.polynomial(self.left_shoulder)
            reward = left + self.shoulder_leakage * (self.polynomial(action)[0] - left)
        elif self.shoulders and action > self.right_shoulder:
            right = self.polynomial(self.right_shoulder)
            reward = right + self.shoulder_leakage * (self.polynomial(action)[0] - right)
        else:
            reward = self.polynomial(action)[0]
        reward += np.random.normal(0, self.std_deviation)

        self.pulls += 1  # Fixed double increment bug
        if self.dynamic_rate is not None and self.pulls % self.dynamic_rate == 0:
            if self.seed is not None:
                self.seed += 1
            self.__draw_polynomial()

        return np.zeros((1,), dtype=np.float32), reward, False, False, {"coef": self.polynomial.coef}
