import pytest
import gymnasium as gym
import numpy as np
import buffalo_gym.envs.fatiguebandit_gym

def test_fatigue_bandit_initialization():
    """Test if the Fatigue Bandit initializes correctly."""
    env = gym.make("FatigueBandit-v0", arms=5, base_mean=10.0, mean_variability=2.0,
                   fatigue_rate=1.0, recovery_rate=0.5, min_reward=0.0, reward_std=1.0, seed=42)

    env.reset()
    assert env.arms == 5
    assert env.base_mean == 10.0
    assert env.mean_variability == 2.0
    assert env.fatigue_rate == 1.0
    assert env.recovery_rate == 0.5
    assert env.min_reward == 0.0
    assert env.reward_std == 1.0
    assert len(env.effective_means) == 5


def test_fatigue_bandit_step():
    """Test if stepping in the environment updates the means correctly."""
    env = gym.make("FatigueBandit-v0", arms=3, base_mean=10.0, mean_variability=0,
                   fatigue_rate=1.0, recovery_rate=0.5, min_reward=0.0, reward_std=1.0, seed=42)

    env.reset()
    obs, reward, done, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert not done
    assert not truncated
    assert len(info["effective_means"]) == 3
    assert info["effective_means"][0] <= 9.0  # Fatigue should have reduced it


def test_fatigue_bandit_recovery():
    """Test if arms recover correctly over time when not selected."""
    env = gym.make("FatigueBandit-v0", arms=2, base_mean=10.0, mean_variability=0,
                   fatigue_rate=1.0, recovery_rate=0.5, min_reward=0.0, reward_std=1.0, seed=42)
    env.reset()
    env.step(0)  # Fatigue arm 0
    env.step(1)  # Fatigue arm 1, arm 0 should recover slightly
    assert env.effective_means[0] > (10.0 - 1.0)


def test_fatigue_bandit_reset():
    """Test if resetting the environment restores initial conditions."""
    env = gym.make("FatigueBandit-v0", arms=4, seed=42)
    env.reset()
    env.step(2)
    env.reset()
    assert np.allclose(env.effective_means, env.max_means)


def test_fatigue_bandit_invalid_action():
    """Test if the environment correctly handles invalid actions."""
    env = gym.make("FatigueBandit-v0", arms=3, seed=42)
    env.reset()
    with pytest.raises(AssertionError):
        env.step(5)  # Invalid arm index
