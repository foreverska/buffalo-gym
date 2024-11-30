from buffalo_gym.utils import *

def test_utils_mab():
    optimal_q = mab_optimal_q(np.array([[1., 0.]]), 0.77)
    assert np.allclose(optimal_q, [4.347, 3.347], atol=0.001)

    optimal_q = mab_optimal_q(np.array([[1., 0., 0.],
                                        [2., 1., 0.]]), 0.77)
    assert np.allclose(optimal_q, np.array([[6.021, 5.021, 5.021],
                                            [7.021, 6.021, 5.021]]), atol=0.001)
