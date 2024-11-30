import numpy as np


def mab_optimal_q(centers: np.ndarray, gamma: float) -> np.ndarray:
    """
    Computes the optimal Q values for a Multi-armed Bandit, assumes all states are equally likely
    :param centers: 2D array of center value of the reward distribution for each arm in each state
    :param gamma: discount factor
    :return: Optimal Q values
    """
    v = [np.max(centers[i, :]) for i in range(centers.shape[0])]
    discounted_v = (gamma * np.mean(v))/(1-gamma)
    return np.array(centers + discounted_v)
