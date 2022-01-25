""" Partition functions """

from abc import ABC, abstractmethod

import numpy as np

# Boltzmann constant in eV/K
KB = 8.617333262145e-5

# Planck's constant in eV/THz
H = 4.135667696e-3


def calc_beta(temperature):
    """
    Converts a temperature array into a beta=1/(KB*temperature) array.

    Parameters
    ----------
    temperature : array_like
        Temperature in K.

    Returns
    -------
    beta : array_like
        Beta array in eV^-1.
    """
    beta = np.divide(1., KB * temperature, where=temperature > 0,
                     out=np.inf * np.ones(temperature.shape))

    return beta


def calc_exponential(observable, temperature, out=0.0):
    """
    Converts a temperature array into a beta=1/(KB*temperature) array.

    Parameters
    ----------
    temperature : array_like
        Temperature in K.

    Returns
    -------
    beta : array_like
        Beta array in eV^-1.
    """
    beta = calc_beta(temperature)[None, :]
    output = out * np.multiply(np.ones_like(observable), np.ones_like(beta))
    exponent = np.multiply(observable, beta, where=observable > 0, out=output)
    exponential = np.exp(-exponent)

    return exponential


def calc_geometric_mean(array):
    """
    Calculates the geometric mean of the input array.

    Parameters
    ----------
    array : array_like
        Input array.

    Returns
    -------
    geometric_mean : array_like
        Geometric mean of the input array.
    """
    geometric_mean = np.exp(np.log(array).mean(axis=1))

    return geometric_mean


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


class ElectronicPF(PartitionFunction):
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

    def __init__(self, potential_energy, symmetry_order, spin_multiplicity):
        self.potential_energy = potential_energy
        self.symmetry_order = symmetry_order
        self.spin_multiplicity = spin_multiplicity

        self.relative_energy = potential_energy - np.amin(potential_energy)

    def calc_part_func(self, temperature):
        """
        Calculates the electronic canonical partition function in the given
        temperature range.

        Parameters
        ----------
        temperature : array_like
            Temperature range in K.

        Returns
        -------
        partition_function: array_like
            Partition function.
        """
        partition_function = calc_exponential(self.relative_energy[:, None],
                                              temperature)
        partition_function /= self.symmetry_order[:, None]
        partition_function *= self.spin_multiplicity[:, None]

        return partition_function


class ClassicalHarmonicPF(PartitionFunction):
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
        self.n_vib = frequencies.shape[1]  # Number of vibrational modes

    def calc_part_func(self, temperature):
        """
        Calculates the classical vibrational partition function in the given
        temperature range.

        Parameters
        ----------
        temperature : array_like
            Temperature range in K.

        Returns
        -------
        partition_function: array_like
            Partition function.
        """

        frequencies_gmean = calc_geometric_mean(self.frequencies)

        partition_function = np.outer(np.power(frequencies_gmean, -self.n_vib),
                                      np.ones_like(temperature))

        return partition_function


class QuantumHarmonicPF(PartitionFunction):
    """
    Represents the canonical vibrational partition function in the quantum
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

    def calc_part_func(self, temperature):
        """
        Calculates the quantum vibrational partition function in the given
        temperature range.

        Parameters
        ----------
        temperature : array_like
            Temperature range in K.

        Returns
        -------
        partition_function: array_like
            Partition function.
        """
        exponential = calc_exponential(0.5 * H * self.frequencies[:, :, None],
                                       temperature)
        partition_function = np.prod(exponential / (1. - exponential**2), axis=1)

        return partition_function
