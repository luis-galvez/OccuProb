""" Utility functions. """

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

# Boltzmann constant in eV/K
KB = 8.617333262145e-5


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


def calc_exponent(energy, temperature):
    """
    Calculates the exponent energy/(KB*temperature) for the given energy and
    temperature arrays.

    Parameters
    ----------
    energy : :obj:`numpy.ndarray`
        A 1D array of size N containing the energy values in eV.
    temperature : :obj:`numpy.ndarray`
        A 1D array of size M containing the temperature values in K.

    Returns
    -------
    exponential : :obj:`numpy.ndarray`
        Calculated exponent stored in a 2D array of shape (N, M)
    """
    if energy.ndim == 1:
        energy = energy[:, None]

    beta = calc_beta(temperature)[None, :]
    output = np.multiply(np.zeros_like(energy), np.zeros_like(beta))
    exponent = np.multiply(energy, beta, where=energy != 0, out=output)

    return exponent


def calc_geometric_mean(in_array):
    """
    Calculates the geometric mean of a 2D input array in_array along the second axis.

    Parameters
    ----------
    in_array : :obj:`numpy.ndarray`
        Input 2D array of shape (N, D).

    Returns
    -------
    geometric_mean : :obj:`numpy.ndarray`
        A 1D array of size N contaning the geometric mean of the input.
    """
    geometric_mean = np.exp(np.mean(np.log(in_array, where=in_array > 0,
                                           out=-np.inf * np.ones_like(in_array)),
                                    axis=1))

    return geometric_mean


def compare_numpy_dictionaries(dict1, dict2):
    """ Checks if two dictionaries containing numpy arrays as values are equal.
    """

    if dict1.keys() != dict2.keys():
        return False

    return all(np.allclose(dict1[key], dict2[key]) for key in dict1)
