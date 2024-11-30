from typing import Any, TypeVar, SupportsFloat
import random

import gymnasium as gym
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BuffaloTrailEnv(gym.Env):
    """
    Multi-armed bandit environment with a secret sequence to trigger max reward.  Tests temporal memory/reasoning.
    """
    metadata = {'render_modes': []}

    def __draw_arms(self):
        """
        Draw new arms
        """
        self.rng = np.random.default_rng(self.seed)
        self.offsets = np.random.uniform(self.min_suboptimal_mean, self.max_suboptimal_mean,
                                         size=(1, self.states, self.arms))

        self.stds = []
        for state in range(self.states):
            optimal_arms = self.rng.choice(range(self.arms), self.optimal_arms, replace=False)
            for arm in optimal_arms:
                self.offsets[0, state, arm] = self.optimal_mean
            self.stds.append([self.optimal_std if arm in optimal_arms else self.suboptimal_std
                              for arm in range(self.arms)])

    def __draw_sequence(self):
        self.rng = np.random.default_rng(self.seed)

        if self.force_aliasing:
            if self.sequence_length < 4:
                raise RuntimeError("Cannot alias bandit environment with sequence_length < 4")
            starting = np.random.choice(range(self.arms))
            alias = np.random.choice(range(self.arms))
            self.goal_sequence = [starting, alias, starting, alias + 1]
            for i in range(self.sequence_length - 4):
                self.goal_sequence.append(np.random.choice(range(self.arms)))
        else:
            self.goal_sequence = [np.random.choice(range(self.arms)) for _ in range(self.sequence_length)]

    def __init__(self, arms: int = 10, optimal_arms: int | list[int] = 1,
                 sequence_length: int = 5, force_aliasing: bool = False, goal_reward: float = 100.0,
                 dynamic_rate: int | None = None, goal_rate: int | None = None,
                 seed: int | None = None, optimal_mean: float = 10, optimal_std: float = 1,
                 min_suboptimal_mean: float = 0, max_suboptimal_mean: float = 5,
                 suboptimal_std: float = 1):
        """
        Stateful Multi-armed bandit environment with k arms and n states
        :param arms: number of arms
        :param optimal_arms: number of optimal arms or list of optimal orms in each state
        :param sequence_length: the length of the secret sequence
        :param force_aliasing: force the aliasing of the bandit environment
        :param goal_reward: the reward of the hidden sequence
        :param dynamic_rate: number of steps between drawing new arm means, None means no dynamic rate
        :param goal_rate: number of steps between drawing new secret sequence, None means goal does not change
        :param seed: random seed
        :param optimal_mean: mean of optimal arms
        :param optimal_std: std of optimal arms
        :param min_suboptimal_mean: min mean of suboptimal arms
        :param max_suboptimal_mean: max mean of suboptimal arms
        :param suboptimal_std: std of suboptimal arms
        """
        self.arms = arms
        self.states = arms
        self.dynamic_rate = dynamic_rate
        self.goal_rate = goal_rate
        self.initial_seed = seed
        self.seed = seed
        self.optimal_mean = optimal_mean
        self.optimal_std = optimal_std
        self.min_suboptimal_mean = min_suboptimal_mean
        self.max_suboptimal_mean = max_suboptimal_mean
        self.suboptimal_std = suboptimal_std
        self.force_aliasing = force_aliasing
        self.sequence_length = sequence_length
        self.goal_reward = goal_reward

        if optimal_arms is list and len(optimal_arms) != self.arms:
            raise ValueError("Optimal arms list must have equal number of arms")
        self.optimal_arms = optimal_arms

        self.action_space = gym.spaces.Discrete(arms)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.pulls = 0
        self.ssr = 0
        self.state = 0

        self.visited = []

        self.__draw_arms()
        self.__draw_sequence()

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

        self.seed = self.initial_seed
        self.pulls = 0
        self.ssr = 0
        self.state = 0
        self.visited = []

        self.__draw_arms()
        self.__draw_sequence()

        return np.zeros((1,), dtype=np.float32), {'goal': self.goal_sequence, 'offsets': self.offsets}

    def step(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Steps the environment
        :param action: arm to pull
        :return: observation, reward, done, term, info
        """
        reward = self.rng.normal(self.offsets[0][self.state][action], self.stds[self.state][action], 1)[0]

        self.state = action
        self.visited.append(self.state)
        if len(self.visited) > len(self.goal_sequence):
            self.visited.pop(0)
        if self.visited == self.goal_sequence:
            reward = self.goal_reward

        self.ssr += 1
        self.pulls += 1
        if self.dynamic_rate is not None and self.pulls % self.dynamic_rate == 0:
            if self.seed is not None:
                self.seed += 1
            self.__draw_arms()

        if self.goal_rate is not None and self.pulls % self.goal_rate == 0:
            if self.seed is not None:
                self.seed += 1
            self.__draw_sequence()

        return np.ones((1,), dtype=np.float32)*self.state, reward, False, False, {'goal': self.goal_sequence,
                                                                                  'offsets': self.offsets}

