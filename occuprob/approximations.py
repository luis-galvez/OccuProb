""" Superposition approximations to the PES """

from abc import ABC, abstractmethod
import numpy as np


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
        age of the person

    Methods
    -------
    calc_probability(temperature):
        Calculates the occupation probability in the temperature range provided.
    calc_heat_capacity(temperature):
        Calculates the canonical heat capacity for the given temperature range.
    calc_ensemble_average(temperature, observable):
        Calculates the ensemble average for the given observable in the provided
        temperature range.
    """

    def __init__(self, potential_energy, frequencies, symmetry_order):
        self.potential_energy = potential_energy
        self.frequencies = frequencies
        self.symmetry_order = symmetry_order

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

    @abstractmethod
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

    @abstractmethod
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
        age of the person

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
