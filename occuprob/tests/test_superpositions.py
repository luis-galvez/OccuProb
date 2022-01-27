"""
Unit and regression test for the occuprob package.
"""

import pytest

import numpy as np

from occuprob.superpositions import ClassicalHarmonicSA, QuantumHarmonicSA


def test_classical_harmonic():
    """Tests the asymptotic behaviour of the occupation probability for a
    ficticious landscape with two minima. The occupation probability is
    calculated using the corresponding canonical partition funcion under
    the classical harmonic superposition approximation"""

    energy = np.array([0.0, 0.001])
    frequencies = np.array([[1., 1., 1.], [2., 1., 1.]])
    symmetry = np.array([3., 1.])
    temperature = np.array([0., np.inf])
    landscape = ClassicalHarmonicSA(energy, frequencies, symmetry)

    expected_prob = np.array([[1.0, 0.4], [0.0, 0.6]])
    calculated_prob = landscape.calc_probability(temperature)

    print(calculated_prob)

    assert pytest.approx(calculated_prob) == expected_prob


def test_quantum_harmonic():
    """Tests the asymptotic behaviour of the occupation probability for a
    ficticious landscape with two minima. The occupation probability is
    calculated using the corresponding canonical partition funcion under
    a quantum harmonic superposition approximation"""

    energy = np.array([0.0, 0.001])
    frequencies = np.array([[1.0, 1.0], [3.0, 1.0]])
    symmetry = np.array([1.0, 1.0])
    multiplicity = np.array([1.0, 1.0])
    temperature = np.array([0.0, 1.0e4])
    landscape = QuantumHarmonicSA(energy, frequencies, symmetry, multiplicity)

    expected_prob = np.array([[1.0, 0.75], [0.0, 0.25]])
    calculated_prob = landscape.calc_probability(temperature)

    assert pytest.approx(calculated_prob, abs=0.001) == expected_prob
