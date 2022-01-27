""" Partition functions """

# MIT License

# Copyright (c) 2021-2022 Luis Gálvez

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
    temperature : :obj:`numpy.ndarray`
        A 1D array of size M containing the temperature values in K.

    Returns
    -------
    beta : :obj:`numpy.ndarray`
        A 1D array of size M containing the values of beta in eV^-1.
    """
    beta = np.divide(1., KB * temperature, where=temperature > 0,
                     out=np.inf * np.ones(temperature.shape))

    return beta


def calc_exponential(energy, temperature):
    """
    Calculates the exponential of the form exp(-energy/(KB*temperature)) for
    the given energy and temperature arrays.

    Parameters
    ----------
    energy : :obj:`numpy.ndarray`
        A 1D array of size N containing the energy values in eV.
    temperature : :obj:`numpy.ndarray`
        A 1D array of size M containing the temperature values in K.

    Returns
    -------
    exponential : :obj:`numpy.ndarray`
        Calculated exponential stored in a 2D array of shape (N, M)
    """
    beta = calc_beta(temperature)[None, :]
    output = np.multiply(np.zeros_like(energy), np.zeros_like(beta))
    exponent = np.multiply(energy, beta, where=energy > 0, out=output)
    exponential = np.exp(-exponent)

    return exponential


def calc_geometric_mean(input_array):
    """
    Calculates the geometric mean of the input array.

    Parameters
    ----------
    input_array : :obj:`numpy.ndarray`
        Input 2D array of shape (N, D).

    Returns
    -------
    geometric_mean : :obj:`numpy.ndarray`
        A 1D array of size N contaning the geometric mean of the input.
    """
    geometric_mean = np.exp(np.log(input_array).mean(axis=1))

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
    potential_energy : :obj:`numpy.ndarray`
        A 1D array containing the energy values (in eV) of each of the N minima.
    spin_multiplicity : :obj:`numpy.ndarray`
        A 1D array containing the spin multiplicity corresponding to each of
        the N minima.

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
        temperature : :obj:`numpy.ndarray`
            A 1D array of size M containing the temperature values in K.

        Returns
        -------
        partition_function: :obj:`numpy.ndarray`
            A 2D array of shape (N, M) contaning the calculated partition
            functions for each of the N minima at every temperature value.
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
    frequencies : :obj:`numpy.ndarray`
        A 2D array of shape (N, D) containing the D frequency values (in THz) of
        each of the N minima.

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
        temperature : :obj:`numpy.ndarray`
            A 1D array of size M containing the temperature values in K.

        Returns
        -------
        partition_function: :obj:`numpy.ndarray`
            A 2D array of shape (N, M) contaning the calculated partition
            functions for each of the N minima at every temperature value.
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
    frequencies : :obj:`numpy.ndarray`
        A 2D array of shape (N, D) containing the D frequency values (in THz) of
        each of the N minima.

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
        temperature : :obj:`numpy.ndarray`
            A 1D array of size M containing the temperature values in K.

        Returns
        -------
        partition_function: :obj:`numpy.ndarray`
            A 2D array of shape (N, M) contaning the calculated partition
            functions for each of the N minima at every temperature value.
        """
        exponential = calc_exponential(0.5 * H * self.frequencies[:, :, None],
                                       temperature)
        partition_function = np.prod(exponential / (1. - exponential**2),
                                     where=temperature > 0, axis=1,
                                     out=np.prod(exponential, axis=1))

        return partition_function
