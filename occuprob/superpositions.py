""" Superposition approximations to the PES """

from abc import ABC, abstractmethod

import numpy as np

import occuprob.partitionfunctions as pf


class SuperpositionApproximation(ABC):
    """
    An abstract class that represents a superposition approximation to the PES.

    ...

    Attributes
    ----------
    potential_energy : array_like
        Potential energy values corresponding to each minima.
    frequencies : array_like
        Frequency values corresponding to each minima.
    symmetry_order : array_like
        Order of the point group symmetry of each minima.

    Methods
    -------
    calc_probability(temperature):
        Calculates the occupation probability in the temperature range provided.
    calc_ensemble_average(temperature, observable):
        Calculates the ensemble average for the given observable in the provided
        temperature range.
    calc_heat_capacity(temperature):
        Calculates the canonical heat capacity for the given temperature range.
    """

    @abstractmethod
    def calc_partition_functions(self, temperature):
        """ Calculates the partition functions for each minima. """

    def calc_probability(self, temperature):
        """
        Calculates the occupation probability in the temperature range provided.

        Parameters
        ----------
        temperature : array_like
            Temperature range.

        Returns
        -------
        occupation_probability : array_like
            Occupation probability.
        """
        partition_functions = self.calc_partition_functions(temperature)

        total_partition_function = np.sum(partition_functions, axis=0)
        occupation_probability = partition_functions / total_partition_function

        return occupation_probability

    def calc_ensemble_average(self, temperature, observable):
        """
        Calculates the ensemble average for the given observable in the provided
        temperature range.

        Parameters
        ----------
        temperature : array_like
            Temperature range.
        observable : array_like
            Temperature range.

        Returns
        -------
        calc_ensemble_average : array_like
            Occupation probability.
        """

        probability = self.calc_probability(temperature)
        ensemble_average = np.sum(observable[:, None] * probability, axis=0)

        return ensemble_average

    def calc_heat_capacity(self, temperature):
        """
        Calculates the canonical heat capacity for the given temperature range.

        Parameters
        ----------
        temperature : array_like
            Temperature range.

        Returns
        -------
        heat_capacity : array_like
            Canonical heat capacity.
        """

        heat_capacity = np.exp(self.calc_probability(temperature))

        return heat_capacity


class ClassicalHarmonicSuperposition(SuperpositionApproximation):
    """
    A class that represents a classical harmonic superposition approximation
    to the PES.

    ...

    Attributes
    ----------
    potential_energy : array_like
        Potential energy values corresponding to each minima.
    frequencies : array_like
        Frequency values corresponding to each minima.
    symmetry_order : array_like
        Order of the point group symmetry of each minima.

    Methods
    -------
    calc_probability(temperature):
        Calculates the occupation probability in the temperature range provided.
    calc_heat_capacity(temperature):
        Calculates the canonical heat capacity for the given temperature range.
    calc_average_observable(temperature, observable):
        Calculates the ensemble average for the given observable in the provided
        temperature range.
    """
    def __init__(self, energy, frequencies, symmetry_order):
        self.energy = energy
        self.global_minimum_index = np.argmin(self.energy)
        self.frequencies = frequencies
        self.symmetry_order = symmetry_order

        # Partition functions
        self.electronic_pf = pf.Electronic(self.energy, 1.0)
        self.vibrational_pf = pf.ClassicalHarmonic(self.frequencies)

    def calc_partition_functions(self, temperature):
        """ Calculates the partition function for each minima. """

        partition_functions = (self.electronic_pf.calc_part_func(temperature) *
                               self.vibrational_pf.calc_part_func(temperature))

        return partition_functions
