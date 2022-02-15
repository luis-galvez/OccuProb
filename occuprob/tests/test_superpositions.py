"""
Unit and regression tests for the occuprob.superposition module.
"""

import pytest

import numpy as np

from occuprob.superpositions import SuperpositionApproximation
from occuprob.partitionfunctions import ElectronicPF
from occuprob.partitionfunctions import ClassicalHarmonicPF
from occuprob.partitionfunctions import QuantumHarmonicPF
from occuprob.partitionfunctions import RotationalPF


def test_empty_sa():
    """Tests instance of the SuperpositionApproximation class where no
    partition functions are included."""

    electronic_sa = SuperpositionApproximation()

    temperature = np.array([0.])
    heat_capacity = electronic_sa.calc_probability(temperature)

    assert heat_capacity is None


def test_electronic_only():
    """Tests the asymptotic behaviour of the occupation probability for a
    ficticious landscape with two minima. The occupation probability is
    calculated using the corresponding canonical eletronic-only partition
    function"""

    potential_energy = np.array([0.0, 0.1])
    multiplicity = np.ones_like(potential_energy)

    partition_functions = [ElectronicPF(potential_energy, multiplicity)]
    electronic_sa = SuperpositionApproximation()
    electronic_sa.add_partition_functions(partition_functions)

    temperature = np.array([0., np.inf])
    expected_prob = np.array([[1.0, 0.5], [0.0, 0.5]])
    calculated_prob = electronic_sa.calc_probability(temperature)

    assert pytest.approx(calculated_prob) == expected_prob


def test_classical_harmonic():
    """Tests the asymptotic behaviour of the occupation probability for a
    ficticious landscape with two minima. The occupation probability is
    calculated using the corresponding canonical partition function under
    the classical harmonic superposition approximation"""

    potential_energy = np.array([0.0, 0.1])
    frequencies = np.array([[1., 1., 1.], [3., 1., 1.]])
    multiplicity = np.ones_like(potential_energy)

    partition_functions = [ElectronicPF(potential_energy, multiplicity),
                           ClassicalHarmonicPF(frequencies)]
    classical_harmonic_sa = SuperpositionApproximation()
    classical_harmonic_sa.add_partition_functions(partition_functions)

    temperature = np.array([0., np.inf])
    expected_prob = np.array([[1.0, 0.75], [0.0, 0.25]])
    calculated_prob = classical_harmonic_sa.calc_probability(temperature)

    assert pytest.approx(calculated_prob) == expected_prob


def test_quantum_harmonic():
    """Tests the asymptotic behaviour of the occupation probability for a
    ficticious landscape with two minima. The occupation probability is
    calculated using the corresponding canonical partition function under
    a quantum harmonic superposition approximation"""

    potential_energy = np.array([0.0, 0.1])
    frequencies = np.array([[1.0, 1.0], [3.0, 1.0]])
    multiplicity = np.array([1.0, 1.0])
    temperature = np.array([0.0, 1.0e5])

    partition_functions = [ElectronicPF(potential_energy, multiplicity),
                           QuantumHarmonicPF(frequencies)]
    quantum_harmonic_sa = SuperpositionApproximation()
    quantum_harmonic_sa.add_partition_functions(partition_functions)

    expected_prob = np.array([[1.0, 0.75], [0.0, 0.25]])
    calculated_prob = quantum_harmonic_sa.calc_probability(temperature)

    assert pytest.approx(calculated_prob, abs=0.1) == expected_prob


def test_classical_harmonic_rot():
    """Tests the asymptotic behaviour of the occupation probability for a
    ficticious landscape with two minima. The occupation probability is
    calculated using the corresponding canonical partition function under
    the classical harmonic superposition approximation including the rotational
    degrees of freedom."""

    potential_energy = np.array([0.0, 0.1])
    frequencies = np.array([[1., 1., 1.], [3., 1., 1.]])
    multiplicity = np.ones_like(potential_energy)
    symmetry_order = np.array([2., 1.])
    moments = np.array([[1., 1., 0.], [1., 1., 0.]])

    partition_functions = [ElectronicPF(potential_energy, multiplicity),
                           ClassicalHarmonicPF(frequencies),
                           RotationalPF(symmetry_order, moments)]
    classical_harmonic_sa = SuperpositionApproximation()
    classical_harmonic_sa.add_partition_functions(partition_functions)

    temperature = np.array([0., np.inf])
    expected_prob = np.array([[1.0, 0.6], [0.0, 0.4]])
    calculated_prob = classical_harmonic_sa.calc_probability(temperature)

    assert pytest.approx(calculated_prob) == expected_prob


def test_quantum_harmonic_rot():
    """Tests the asymptotic behaviour of the occupation probability for a
    ficticious landscape with two minima. The occupation probability is
    calculated using the corresponding canonical partition function under
    a quantum harmonic superposition approximation including the rotational
    degrees of freedom."""

    potential_energy = np.array([0.0, 0.1])
    frequencies = np.array([[1.0, 1.0], [3.0, 1.0]])
    multiplicity = np.array([1.0, 1.0])
    temperature = np.array([0.0, 1.0e5])
    symmetry_order = np.array([2., 1.])
    moments = np.array([[1., 1., 0.], [1., 1., 0.]])

    partition_functions = [ElectronicPF(potential_energy, multiplicity),
                           QuantumHarmonicPF(frequencies),
                           RotationalPF(symmetry_order, moments)]
    quantum_harmonic_sa = SuperpositionApproximation()
    quantum_harmonic_sa.add_partition_functions(partition_functions)

    expected_prob = np.array([[1.0, 0.6], [0.0, 0.4]])
    calculated_prob = quantum_harmonic_sa.calc_probability(temperature)

    print(calculated_prob)

    assert pytest.approx(calculated_prob, abs=0.1) == expected_prob
