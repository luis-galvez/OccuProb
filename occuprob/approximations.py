""" Superposition approximations to the PES """

from abc import ABC, abstractmethod
import numpy as np

# Boltzmann constant in eV/K
KB = 8.617333262145e-5


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

    def __init__(self, potential_energy, frequencies, symmetry_order):
        self.potential_energy = potential_energy
        self.relative_energy = potential_energy - np.amin(potential_energy)
        self.global_minimum_index = np.argmin(self.potential_energy)
        self.frequencies = frequencies
        self.symmetry_order = symmetry_order
        self.degrees_of_freedom = frequencies.shape[0]

    @abstractmethod
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

        heat_capacity = self.degrees_of_freedom * KB / temperature

        return heat_capacity


class ClassicalHarmonic(SuperpositionApproximation):
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
        beta = np.divide(1., KB * temperature, where=temperature > 0,
                         out=np.inf * np.ones(temperature.shape))
        exponent = np.multiply(self.relative_energy[:, None], beta[None, :],
                               where=self.relative_energy[:, None] > 0,
                               out=np.zeros((self.relative_energy.size,
                                             temperature.size)))
        partition_functions = np.exp(-exponent)
        total_partition_function = np.sum(partition_functions, axis=0)
        occupation_probability = partition_functions / total_partition_function

        return occupation_probability
