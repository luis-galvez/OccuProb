""" Superposition approximations to the PES """

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

import numpy as np

import occuprob.partitionfunctions as pf


class SuperpositionApproximation():
    """
    Represents a superposition approximation of the PES.

    ...

    Methods
    -------
    calc_partition_functions(temperature):
        Calculates the individual partition function contributions of each
        geometrically unique isomer
    calc_probability(temperature):
        Calculates the occupation probability in the temperature range provided.
    calc_ensemble_average(temperature, observable):
        Calculates the ensemble average for the given observable in the provided
        temperature range.
    calc_heat_capacity(temperature):
        Calculates the canonical heat capacity for the given temperature range.
    """

    def __init__(self, potential_energy, frequencies, symmetry_order,
                 spin_multiplicity):
        self.potential_energy = potential_energy
        self.frequencies = frequencies
        self.symmetry_order = symmetry_order
        self.spin_multiplicity = spin_multiplicity

        self.partition_functions = []

    def calc_partition_functions(self, temperature):
        """ Calculates the individual partition functions of each geometrically
        unique isomer.

        Parameters
        ----------
        temperature : :obj:`numpy.ndarray`
            A 1D array of size M containing the temperature values in K.

        Returns
        -------
        partition_functions : :obj:`numpy.ndarray`
            A 2D array of shape (N, M) containing the individual partition
            function contributions of each of the N minima.
        """
        if not self.partition_functions:
            print("You must include at least one partition function.")
            return None

        contributions = [partition_function.calc_part_func(temperature) for
                         partition_function in self.partition_functions]
        partition_functions = np.prod(np.stack(contributions), axis=0)

        return partition_functions

    def calc_probability(self, temperature):
        """
        Calculates the occupation probability in the temperature range provided.

        Parameters
        ----------
        temperature : :obj:`numpy.ndarray`
            A 1D array of size M containing the temperature values in K.

        Returns
        -------
        occupation_probability : :obj:`numpy.ndarray`
            A 2D array of shape (N, M) containing the occupation probability of
            each of the N minima.
        """
        # Calculates the individual partition function contributions
        partition_functions = self.calc_partition_functions(temperature)

        # The total partition function is the sum of all the individual
        # contributions of each geometrically unique isomer
        total_partition_function = np.sum(partition_functions, axis=0)

        # The probability of finding the system in some isomer is calculated
        # as the quotient of its individual partition function divided by the
        # total partition function
        occupation_probability = partition_functions / total_partition_function

        return occupation_probability

    def calc_ensemble_average(self, temperature, observable):
        """
        Calculates the ensemble average for the given observable in the provided
        temperature range.

        Parameters
        ----------
        temperature : :obj:`numpy.ndarray`
            A 1D array of size M containing the temperature values in K.
        observable : :obj:`numpy.ndarray`
            A 1D array of size N containing the input observable.

        Returns
        -------
        ensemble_average : :obj:`numpy.ndarray`
            A 1D array of shape M containing the ensemble average of the input
            observable at every temperature provided.
        """

        # The ensemble average of a given observable is calculated as a
        # weighted sum using the occupation probabilities of each unique isomer
        # as the weights
        probability = self.calc_probability(temperature)
        ensemble_average = np.sum(observable[:, None] * probability, axis=0)

        return ensemble_average

    def calc_heat_capacity(self, temperature):
        """
        Calculates the canonical heat capacity for the given temperature range.

        Parameters
        ----------
        temperature : :obj:`numpy.ndarray`
            A 1D array of size M containing the temperature values in K.

        Returns
        -------
        heat_capacity : :obj:`numpy.ndarray`
            A 1D array of shape M containing the heat capacity of the system.
        """

        heat_capacity = np.exp(self.calc_probability(temperature))

        return heat_capacity


class ClassicalHarmonicSA(SuperpositionApproximation):
    """
    Represents a classical harmonic superposition approximation of the PES.

    ...

    Attributes
    ----------
    potential_energy : :obj:`numpy.ndarray`
        A 1D array containing the energy values (in eV) of each of the N minima.
    frequencies : :obj:`numpy.ndarray`
        A 2D array of shape (N, D) containing the D frequency values (in THz) of
        each of the N minima.
    symmetry_order : :obj:`numpy.ndarray`
        A 1D array of size M containing the order of the point group symmetry
        corresponding to each minimum.

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
    def __init__(self, potential_energy, frequencies, symmetry_order):
        super().__init__(potential_energy, frequencies, symmetry_order,
                         np.ones_like(potential_energy))

        # Partition functions
        self.partition_functions = [pf.ElectronicPF(self.potential_energy,
                                                    self.symmetry_order,
                                                    self.spin_multiplicity),
                                    pf.ClassicalHarmonicPF(self.frequencies)]


class QuantumHarmonicSA(SuperpositionApproximation):
    """
    Represents a classical harmonic superposition approximation of the PES.

    ...

    Attributes
    ----------
    potential_energy : :obj:`numpy.ndarray`
        A 1D array containing the energy values (in eV) of each of the N minima.
    frequencies : :obj:`numpy.ndarray`
        A 2D array of shape (N, D) containing the D frequency values (in THz) of
        each of the N minima.
    symmetry_order : :obj:`numpy.ndarray`
        A 1D array of size M containing the order of the point group symmetry
        corresponding to each minimum.

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
    def __init__(self, potential_energy, frequencies, symmetry_order,
                 spin_multiplicity):
        super().__init__(potential_energy, frequencies, symmetry_order,
                         spin_multiplicity)

        # Partition functions
        self.partition_functions = [pf.ElectronicPF(self.potential_energy,
                                                    self.symmetry_order,
                                                    self.spin_multiplicity),
                                    pf.QuantumHarmonicPF(self.frequencies)]
