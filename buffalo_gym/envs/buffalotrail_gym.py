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

    def __init__(self, arms: int = 5, sequence_length: int = 5, force_aliasing: bool = False):
        """
        Multi-armed bandit environment with k arms and n states
        :param arms: number of arms
        :param sequence_length: the length of the secret sequence
        :param force_aliasing: force the alias of the bandit environment
        """
        self.action_space = gym.spaces.Discrete(arms)
        self.observation_space = gym.spaces.Box(low=0, high=arms, shape=(1,), dtype=np.float32)

        self.offsets = np.random.normal(0, arms, (arms, arms))
        self.states = arms
        self.state = 0
        if force_aliasing:
            if sequence_length < 4:
                raise RuntimeError("Cannot alias bandit environment with sequence_length < 4")
            starting = random.choice(range(arms))
            alias = random.choice(range(arms))
            self.goal_sequence = [starting, alias, starting, alias+1]
            for i in range(sequence_length - 4):
                self.goal_sequence.append(random.choice(range(arms)))
        else:
            self.goal_sequence = [random.choice(range(arms)) for _ in range(sequence_length)]
        self.visited = []

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
        self.visited = [0]

        return np.zeros((1,), dtype=np.float32), {'goal': self.goal_sequence}

    def step(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Steps the environment
        :param action: arm to pull
        :return: observation, reward, done, term, info
        """
        self.state = action
        reward = np.random.normal(0, 1, 1)[0] + self.offsets[action, self.state]
        self.visited.append(self.state)
        if len(self.visited) > len(self.goal_sequence):
            self.visited.pop(0)
        if self.visited == self.goal_sequence:
            reward = 100

        return np.ones((1,), dtype=np.float32)*action, reward, False, False, {}
