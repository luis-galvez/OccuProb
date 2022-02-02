""" Module used to create Superposition Approximations to the PES. """

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

    def __init__(self):
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

        if partition_functions is None:
            return None

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
            A 2D array of shape (N, M) containing the input observable values for
            each isomes as a function of the temperature.

        Returns
        -------
        ensemble_average : :obj:`numpy.ndarray`
            A 1D array of shape M containing the ensemble average of the input
            observable at the given temperature range.
        """

        # The ensemble average of a given observable is calculated as a
        # weighted sum using the occupation probabilities of each unique isomer
        # as the weights
        probability = self.calc_probability(temperature)

        if probability is None:
            return None

        weighted_observable = np.multiply(observable, probability,
                                          where=probability > 0,
                                          out=np.zeros((observable.shape[0],
                                                        temperature.size)))
        ensemble_average = np.sum(weighted_observable, axis=0)

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

        part_func_w = [partition_function.calc_part_func_w(temperature) for
                       partition_function in self.partition_functions]

        part_func_v = [partition_function.calc_part_func_v(temperature) for
                       partition_function in self.partition_functions]

        contribution_w = np.sum(np.stack(part_func_w), axis=0)
        contribution_v = np.sum(np.stack(part_func_v), axis=0)

        heat_capacity = (self.calc_ensemble_average(temperature, contribution_v) +
                         self.calc_ensemble_average(temperature, contribution_w**2) -
                         self.calc_ensemble_average(temperature, contribution_w)**2)

        return heat_capacity
