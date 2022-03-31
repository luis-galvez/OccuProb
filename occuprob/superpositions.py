""" Module used to combine partition function contributions and create superposition aproximations to the PES. """

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
    Represents a superposition approximation of the PES for a particular system.
    In this approximation, the partition function of the system is written as the
    sum of the individual contributions from each geometrically unique local
    minimum the PES:

    .. math::
        Z = \\sum_{a=1}^{N} Z_a,

    where :math:`Z_a` is the partition function for the local minimum :math:`a`.

    """

    def __init__(self):
        self.partition_functions = []

    def add_partition_functions(self, partition_functions):
        """ Add partition function contributions.

        Parameters
        ----------
        partition_functions: :obj:`PartitionFunction` or list of :obj:`PartitionFunction`
            Partition function contribution(s) to add.
        """
        try:
            self.partition_functions.extend(partition_functions)
        except TypeError:
            self.partition_functions.append(partition_functions)

    def combine_contributions(self, temperature, combiner, method):
        """ Calculates and combines the contributions of each degree of freedom
        to the partition functions :math:`Z_a` or their derivatives with respect
        to :math:`\\beta` (:math:`W_a` or :math:`V_a`).

        Parameters
        ----------
        temperature : :obj:`numpy.ndarray`
            A 1D array of size M containing the temperature values in K.
        combiner: callable
            Function used to combine the individual contributions. Can be either
            Numpy.sum or Numpy.prod.
        method: string
            Name of the PartitionFunction class method used to compute the
            individual contributions ("calc_func", "calc_func_w", "calc_func_v")

        Returns
        -------
        combined_functions : :obj:`numpy.ndarray`
            A 2D array of shape (N, M) containing the combined contributions for
            each of the N minima.
        """
        try:
            contributions = [getattr(partition_function, method)(temperature) for
                             partition_function in self.partition_functions]
            combined_contributions = combiner(np.stack(contributions), axis=0)

            return combined_contributions

        except ValueError:
            print("You must include at least one partition function.")

    def calc_partition_functions(self, temperature):
        """ Calculates the individual contributions of each local minima to the
        partition function:

        .. math::
            Z_a = Z_{elec,a}Z_{vib,a}Z_{rot,a}\\cdots

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
        partition_functions = self.combine_contributions(temperature, np.prod,
                                                         "calc_part_func")

        return partition_functions

    def calc_probability(self, temperature):
        """
        Calculates the occupation probability in the temperature range provided,
        which is given by:

        .. math::
            P_a = \\frac{Z_a}{Z}

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
        temperature range, which can be computed as:

        .. math::
            \\langle B \\rangle = \\sum_{a=1}^{N} B_a P_a,

        where :math:`P_a` is the occupation probability of minimum :math:`a`.

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
            A 2D array of shape (1, M) containing the ensemble average of the
            input observable at the given temperature range.
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

        return ensemble_average.reshape(1, -1)

    def calc_heat_capacity(self, temperature):
        """
        Calculates the canonical heat capacity for the given temperature range:

        .. math::
            \\frac{C_v}{k_B} &= \\beta^2 \\left[-\\frac{1}{Z^2} \\left(\\sum_a W_a Z_a\\right)^2
                   + \\frac{1}{Z} \\sum_a \\left(W^2_a Z_a + V_a Z_a\\right)\\right] \\\\
                &= -\\left(\\sum_a\\beta W_a P_a\\right)^2
                       + \\sum_a \\beta^2 \\left(W^2_a P_a + V_a P_a\\right) \\\\
                &= -\\langle \\beta W \\rangle^2 + \\langle (\\beta W)^2 \\rangle
                   +\\langle \\beta^2 V \\rangle

        where

        .. math::
            W_a = \\frac{1}{Z_a} \\frac{\\partial Z_a}{\\partial \\beta}
                = W_{elec, a} + W_{vib, a} + W_{rot, a} + \\dots

        and

        .. math::
            V_a = \\frac{\\partial W_a}{\\partial \\beta}
                = V_{elec, a} + V_{vib, a} + V_{rot, a} + \\dots

        Parameters
        ----------
        temperature : :obj:`numpy.ndarray`
            A 1D array of size M containing the temperature values in K.

        Returns
        -------
        heat_capacity : :obj:`numpy.ndarray`
            A 2D array of shape (1, M) containing the heat capacity of the system.
        """
        part_func_d = {d: self.combine_contributions(temperature, np.sum,
                                                     'calc_part_func_' + d)
                       for d in ['w', 'v']}

        heat_capacity = (self.calc_ensemble_average(temperature, part_func_d['v']) +
                         self.calc_ensemble_average(temperature, part_func_d['w']**2) -
                         self.calc_ensemble_average(temperature, part_func_d['w'])**2)

        return heat_capacity
