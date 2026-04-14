import numpy as np
from pbrl_emoa.preference.value_functions import linear_value_function, quadratic_value_function

def test_linear_value_function():
    vf = linear_value_function(np.array([0.5, 0.5]))
    assert np.isclose(vf.value(np.array([0.2, 0.4])), 0.3)

def test_quadratic_value_function():
    vf = quadratic_value_function(np.array([0.5, 0.5]), np.array([0.0, 0.0]))
    assert np.isclose(vf.value(np.array([2.0, 0.0])), 1.0)
