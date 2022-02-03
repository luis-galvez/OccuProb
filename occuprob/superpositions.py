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
    combine_contributions(temperature, method, combiner):
        Calculates and combines the contributions of each degree of freedom
        to the partition functions or their derivatives with respect to Beta
    calc_partition_functions(temperature):
        Calculates the partition function of each local minimum considered.
    calc_probability(temperature):
        Calculates the occupation probability in the temperature range provided.
    calc_ensemble_average(temperature, observable):
        Calculates the ensemble average for the given observable in the provided
        temperature range.
    calc_heat_capacity(temperature):
        Calculates the canonical heat capacity for the given temperature range.
    """

    def __init__(self):
        self.partition_functions = []  # Degrees of freedom to consider

    def combine_contributions(self, temperature, method, combiner):
        """ Calculates and combines the contributions of each degree of freedom
        to the partition functions or their derivatives with respect to Beta.

        Parameters
        ----------
        temperature : :obj:`numpy.ndarray`
            A 1D array of size M containing the temperature values in K.
        method: string
            Name of the PartitionFunction class method used to compute the
            individual contributions ("calc_func", "calc_func_w", "calc_func_v")
        combiner: callable
            Function used to combine the individual contributions. Can be either
            Numpy.sum or Numpy.prod.

        Returns
        -------
        combined_functions : :obj:`numpy.ndarray`
            A 2D array of shape (N, M) containing the combined contributions for
            each of the N minima.
        """
        if not self.partition_functions:
            print("You must include at least one partition function.")
            return None

        contributions = [getattr(partition_function, method)(temperature) for
                         partition_function in self.partition_functions]
        combined_contributions = combiner(np.stack(contributions), axis=0)

        return combined_contributions

    def calc_partition_functions(self, temperature):
        """ Calculates the partition functions of each local minimum considered.

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
        partition_functions = self.combine_contributions(temperature,
                                                         "calc_part_func",
                                                         np.prod)

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
        # contributions of each local minimum considered
        total_partition_function = np.sum(partition_functions, axis=0)

        # The probability of finding the system in each local minimum is
        # calculated by dividing their individual contributions by the total
        # partition function
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
            each local minimum as a function of the temperature.

        Returns
        -------
        ensemble_average : :obj:`numpy.ndarray`
            A 1D array of shape M containing the ensemble average of the input
            observable at the given temperature range.
        """

        # The ensemble average of a given observable is calculated as a
        # weighted sum using the occupation probabilities of each local minimum
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
        part_func_w = self.combine_contributions(temperature, "calc_part_func_w",
                                                 np.sum)
        part_func_v = self.combine_contributions(temperature, "calc_part_func_v",
                                                 np.sum)

        heat_capacity = (self.calc_ensemble_average(temperature, part_func_v) +
                         self.calc_ensemble_average(temperature, part_func_w**2) -
                         self.calc_ensemble_average(temperature, part_func_w)**2)

        return heat_capacity
