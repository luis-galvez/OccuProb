"""
Unit and regression test for the occuprob package.
"""
import pytest

import numpy as np

from occuprob import approximations

def test_classical_harmonic():
    """Tests the asymptotic behaviour of the occupation probability for a
    ficticious landscape with two minima. The occupation probability is
    calculated using the corresponding canonical partition funcion under
    the classical harmonic superposition approximation"""

    energy = [0.0, 0.5]
    frequencies = [[1.0], [1.0]]
    symmetry = [1.0, 1.0]
    temperature = np.array([0.0, np.inf])
    landscape = approximations.ClassicalHarmonic(energy, frequencies, symmetry)

    expected_probability = np.array([[1.0, 0.5], [0.0, 0.5]])

    assert landscape.calc_probability(temperature) == expected_probability
