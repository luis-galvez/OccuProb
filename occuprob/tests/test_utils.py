"""
Unit and regression test for the occuprob.partitionfunctions package.
"""

import pytest

import numpy as np

from occuprob.utils import calc_beta, calc_geometric_mean, calc_exponential


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


def test_calc_exponential():
    """ Unit test for the calc_exponential function."""

    energy = np.array([0., 1.])
    temperature = np.array([0., 11604.5181216])

    expected_exponential = np.array([[1., 1.], [0., 0.367879441171]])
    calculated_exponential = calc_exponential(energy, temperature)
    print(calculated_exponential)

    assert pytest.approx(calculated_exponential) == expected_exponential
