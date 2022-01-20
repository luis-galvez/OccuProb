"""
Unit and regression test for the occuprob package.
"""
import pytest

import numpy as np

from occuprob.approximations import ClassicalHarmonic


def test_classical_harmonic():
    """Tests the asymptotic behaviour of the occupation probability for a
    ficticious landscape with two minima. The occupation probability is
    calculated using the corresponding canonical partition funcion under
    the classical harmonic superposition approximation"""

    energy = np.array([0.0, 0.5])
    frequencies = np.array([[1.0], [1.0]])
    symmetry = np.array([1.0, 1.0])
    temperature = np.array([0.0, np.inf])
    landscape = ClassicalHarmonic(energy, frequencies, symmetry)

    expected_probability = np.array([[1.0, 0.5], [0.0, 0.5]])
    calculated_probability = landscape.calc_probability(temperature)

    assert (calculated_probability == expected_probability).all()
