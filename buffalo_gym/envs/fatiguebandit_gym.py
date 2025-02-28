import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.utils import seeding


class FatigueBanditEnv(gym.Env):
    """
    Fatigue Bandit (Buffalo Name: TiredBuffalo)

    This bandit problem models resource depletion and recovery. Pulling an arm reduces its expected
    reward ("fatigue"), while unused arms gradually recover. Each arm has a unique maximum mean reward,
    requiring the agent to balance immediate rewards against long-term sustainability.

    The Fatigue Bandit is a special case of a **restless bandit problem** (Whittle, 1988) and is conceptually
    related to "Recovering Bandits" (Pike-Burke & Grünewälder, 2019). Like in their framework,
    arms transition dynamically over time, even when not selected. However, unlike general restless bandits,
    our model features a deterministic recovery and decay mechanism with fixed parameters.

    Parameters:
        arms (int): Number of available arms (default: 10).
        base_mean (float): Mean around which arm rewards are initialized (default: 10.0).
        mean_variability (float): Range for randomizing each arm’s max mean (default: 3.0).
        fatigue_rate (float): Decrease in effective mean per pull (default: 1.0).
        recovery_rate (float): Increase in effective mean per step when an arm is not pulled (default: 0.5).
        min_reward (float): Lower bound on an arm's effective mean (default: 0.0).
        reward_std (float): Standard deviation for reward sampling (default: 1.0).
        seed (int, optional): Random seed for reproducibility.

    Usage:
        >>> env = gym.make("FatigueBandit-v0", arms=10, base_mean=10.0, mean_variability=3.0,
        ...                  fatigue_rate=1.0, recovery_rate=0.5, min_reward=0.0,
        ...                  reward_std=1.0, seed=42)
        >>> obs, reward, done, truncated, info = env.step(0)
    """
    metadata = {"render_modes": []}

    def __init__(self, arms=10, base_mean=10.0, mean_variability=3.0, fatigue_rate=1.0,
                 recovery_rate=0.5, min_reward=0.0, reward_std=1.0, seed=None):
        """
        Initializes the Fatigue Bandit environment.
        """
        super(FatigueBanditEnv, self).__init__()
        self.arms = arms
        self.base_mean = base_mean
        self.mean_variability = mean_variability
        self.fatigue_rate = fatigue_rate
        self.recovery_rate = recovery_rate
        self.min_reward = min_reward
        self.reward_std = reward_std

        self._np_random, _ = seeding.np_random(seed)
        self.seed_val = seed

        # Assign each arm a unique max mean reward
        self.max_means = self._np_random.uniform(
            low=self.base_mean - self.mean_variability,
            high=self.base_mean + self.mean_variability,
            size=self.arms
        )

        self.effective_means = self.max_means.copy()

        self.action_space = spaces.Discrete(self.arms)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.step_count = 0

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        if seed is not None:
            self._np_random, _ = seeding.np_random(seed)
        self.effective_means = self.max_means.copy()
        self.step_count = 0
        return np.array([0], dtype=np.float32), {}

    def step(self, action):
        """
        Takes a step in the environment by selecting an arm.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        current_mean = self.effective_means[action]
        reward = self._np_random.normal(loc=current_mean, scale=self.reward_std)

        self.effective_means[action] = max(current_mean - self.fatigue_rate, self.min_reward)

        for arm in range(self.arms):
            if arm != action:
                self.effective_means[arm] = min(
                    self.effective_means[arm] + self.recovery_rate,
                    self.max_means[arm]
                )

        self.step_count += 1
        done = False
        truncated = False
        info = {"effective_means": self.effective_means.copy(), "step_count": self.step_count}
        return np.array([0], dtype=np.float32), reward, done, truncated, info

    def close(self):
        """
        Performs any necessary cleanup when closing the environment.
        """
        pass
