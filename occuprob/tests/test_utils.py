"""
Unit and regression tests for the occuprob.utils module.
"""

import pytest

import numpy as np

from occuprob.utils import calc_beta
from occuprob.utils import calc_geometric_mean
from occuprob.utils import calc_exponent
from occuprob.utils import compare_numpy_dictionaries


def test_calc_beta():
    """ Unit test for the calc_beta function."""

    temperature = np.array([0., 11604.5181216])

    expected_beta = np.array([np.inf, 1.])
    calculated_beta = calc_beta(temperature)

    assert pytest.approx(calculated_beta) == expected_beta


def test_calc_geometric_mean():
    """ Unit test for the calc_geometric_mean function."""

    input_array = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])

    calculated_gmean = calc_geometric_mean(input_array)

    expected_gmean = np.array([0., 1., 2., 3.])
    print(calculated_gmean)

    assert pytest.approx(calculated_gmean) == expected_gmean


def test_calc_exponent():
    """ Unit test for the calc_exponent function."""

    energy = np.array([0., 1.])
    temperature = np.array([0., 11604.518121, 23209.036242])

    expected_exponent = np.array([[0., 0., 0.], [np.inf, 1., 0.5]])
    calculated_exponent = calc_exponent(energy, temperature)

    assert pytest.approx(calculated_exponent) == expected_exponent


def test_compare_numpy_dictionaries():
    """ Unit test for the compare_numpy_dictionaries function. """

    dict1 = {1: np.zeros((2, 2)), 2: np.ones((2, 2))}
    dict2 = {1: np.zeros((2, 2)), 2: np.ones((2, 2))}
    dict3 = {1: np.ones((2, 2)), 2: np.zeros((2, 2))}
    dict4 = {0: np.ones((2, 2)), 1: np.zeros((2, 2))}

    assert compare_numpy_dictionaries(dict1, dict2)
    assert not compare_numpy_dictionaries(dict1, dict3)
    assert not compare_numpy_dictionaries(dict3, dict4)
