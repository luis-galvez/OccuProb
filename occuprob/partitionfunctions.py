""" Partition functions """

from abc import ABC, abstractmethod
import numpy as np

# Boltzmann constant in eV/K
KB = 8.617333262145e-5


class PartitionFunction(ABC):
    """
    An abstract class that represents a partition function.

    ...

    Methods
    -------
    calc_partition_function(temperature):
        Calculates the partition function in the given temperature range.
    """

    @abstractmethod
    def calc_part_func(self, temperature):
        """
        Abstract method to calculate the partition function.
        """


class Electronic(PartitionFunction):
    """
    Represents the canonical electronic partition function.

    ...
    potential_energy : array_like
        Potential energy values corresponding to each minima.
    spin_multiplicity : array_like
        Spin degenerancy of each isomer.

    Methods
    -------
    calc_partition_function(temperature):
        Calculates the canonical electronic partition function in the given
        temperature range.
    """

    def __init__(self, potential_energy, spin_multiplicity):
        self.potential_energy = potential_energy
        self.relative_energy = potential_energy - np.amin(potential_energy)
        self.spin_multiplicity = spin_multiplicity

    def calc_part_func(self, temperature):
        """
        Calculates the electronic canonical partition function in the given
        temperature range.

        Parameters
        ----------
        temperature : array_like
            Temperature range.

        Returns
        -------
        partition_function: array_like
            Partition function.
        """
        beta = np.divide(1., KB * temperature, where=temperature > 0,
                         out=np.inf * np.ones(temperature.shape))
        exponent = np.multiply(self.relative_energy[:, None], beta[None, :],
                               where=self.relative_energy[:, None] > 0,
                               out=np.zeros((self.relative_energy.size,
                                             temperature.size)))
        partition_function = np.exp(-exponent)

        return partition_function


class ClassicalHarmonic(PartitionFunction):
    """
    Represents the canonical vibrational partition function in the classical
    harmonic approximation.

    ...
    frequencies : array_like
        Frequency values corresponding to each minima.

    Methods
    -------
    calc_partition_function(temperature):
        Calculates the vibrational partition function in the given
        temperature range.
    """

    def __init__(self, frequencies):
        self.frequencies = frequencies
        self.degrees_of_freedom = frequencies.shape[0]

    def calc_part_func(self, temperature):
        """
        Calculates the vibrational partition function in the given
        temperature range.

        Parameters
        ----------
        temperature : array_like
            Temperature range.

        Returns
        -------
        partition_function: array_like
            Partition function.
        """
        partition_function = 1.0

        return partition_function
