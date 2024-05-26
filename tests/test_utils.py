from buffalo_gym.utils import *

def test_utils_mab():
    optimal_q = mab_optimal_q(np.array([[1., 0.]]), 0.77)
    assert np.allclose(optimal_q, [4.347, 3.347], atol=0.001)
