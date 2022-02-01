"""
Unit and regression tests for the occuprob.superposition module.
"""

import pytest

import numpy as np

from occuprob.partitionfunctions import ElectronicPF


def test_electronic():
    """ Test the methods in the ElectronicPF class used to calculate the
    derivatives of the partition function with respect to Beta"""

    potential_energy = np.array([0.0, 0.1, 0.2])
    spin_multiplicity = np.array([1., 3., 5.])

    partition_function = ElectronicPF(potential_energy, spin_multiplicity)

    temperature = np.array([0., np.inf])
    expected_part_func = np.array([[1.0, 1.0], [0.0, 3.0], [0.0, 5.0]])
    calculated_part_func = partition_function.calc_part_func(temperature)

    assert (calculated_part_func == expected_part_func).all()


def test_electronic_derivatives():
    """ Test the methods in the ElectronicPF class used to calculate the
    derivatives of the partition function with respect to Beta"""

    potential_energy = np.array([0.0, 0.1, 0.2])
    spin_multiplicity = np.array([1., 3., 5.])

    partition_function = ElectronicPF(potential_energy, spin_multiplicity)

    temperature = np.array([0., np.inf])
    expected_part_func_v = np.zeros((potential_energy.size, temperature.size))
    calculated_part_func_v = partition_function.calc_part_func_v(temperature)

    assert (calculated_part_func_v == expected_part_func_v).all()
