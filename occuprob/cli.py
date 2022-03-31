""" Command-line interface. """

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

import argparse

import numpy as np

from occuprob import io
from occuprob.superpositions import SuperpositionApproximation
from occuprob.partitionfunctions import ElectronicPF
from occuprob.partitionfunctions import ClassicalHarmonicPF
from occuprob.partitionfunctions import QuantumHarmonicPF
from occuprob.partitionfunctions import RotationalPF

from ._version import get_versions


def save_results(results, temperature, outfile):
    """ Store results in a plain text file.

    Parameters
    ----------
    results : :obj:`numpy.ndarray`
        A 2D array shape (N, M) containing the data to be plotted.
    temperature : :obj:`numpy.ndarray`
        A 1D array of size M containing the temperature values in K.
    outfile : string
        Output filename.
    """
    outdata = np.vstack((temperature, results)).T
    np.savetxt(outfile, outdata)


def plot_results(results, results_type, temperature, outfile, size):
    """ Plots results.

    Parameters
    ----------
    results : :obj:`numpy.ndarray`
        A 2D array shape (N, M) containing the data to be plotted.
    results_type : {'P', 'C'}
        Type of data to be plotted: can be 'P' for occupation probability or 'C'
        for heat capacity.
    temperature : :obj:`numpy.ndarray`
        A 1D array of size M containing the temperature values in K.
    outfile : string
        Output filename.
    size : tuple of floats
        Width and height of the output figure, e.g, (8., 6.)
    """
    plot_format = {}

    if results_type == 'P':
        plot_format['labels'] = ['Isomer ' + str(i+1) for i in range(len(results))]
        plot_format['hline_pos'] = [0, 1]
        plot_format['ylims'] = (-0.05, 1.05)
        plot_format['ylabel'] = r'Occupation probability, $P_a(T)$'
    elif results_type == 'C':
        plot_format['ylims'] = (0.0, 2. * np.ceil(results.max() / 2.))
        plot_format['ylabel'] = r'Heat capacity, $C_V/k_B$'

    plot_format['size'] = size

    io.plot_results(results, temperature, outfile, **plot_format)


def main():
    """ Command-line interface """
    description = 'OccuProb: A tool for calculating occupation probabilities' +\
                  ' and ensemble-averaged properties via the superposition approximation.'
    versions = get_versions()
    __version__ = versions['version']

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__}')

    electronic = parser.add_mutually_exclusive_group()
    electronic.add_argument('-e', '-E', action='store_true',
                            help='Add electronic partition function')
    electronic.add_argument('-s', '-S', action='store_true',
                            help='Add electronic partition function (including spin)')
    vibrational = parser.add_mutually_exclusive_group()
    vibrational.add_argument('-c', '-C', action='store_true',
                             help='Add vibrational partition function (classical harmonic)')
    vibrational.add_argument('-q', '-Q', action='store_true',
                             help='Add vibrational partition function (quantum harmonic)')
    parser.add_argument('-r', '-R', action='store_true',
                        help='Add rotational partition function')
    parser.add_argument('in_file', help='Extended XYZ file contaning the list of isomers')
    parser.add_argument('--out_file', help='Output filename prefix')
    parser.add_argument('--plot_format', default='svg',
                        help='Output image files plot_format (default: SVG)')
    parser.add_argument('--min_temp', type=float, default=0.,
                        help='Maximum temperature in K (default: 0)')
    parser.add_argument('--max_temp', type=float, default=500.,
                        help='Maximum temperature in K (default: 500)')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the results and save them as image files')
    parser.add_argument('--size', type=float, nargs=2, default=[8., 6.],
                        help='Width and height of the output image, in inches (default: 8.0 6.0)')
    args = parser.parse_args()

    properties = io.load_properties_from_extxyz(args.in_file)

    # Add the partition functions specified by the user
    partition_functions = []

    if args.e:
        partition_functions.append(ElectronicPF(properties['energy'],
                                                np.ones_like(properties['energy'])))
    if args.s:
        partition_functions.append(ElectronicPF(properties['energy'],
                                                properties['multiplicity']))
    if args.c:
        partition_functions.append(ClassicalHarmonicPF(properties['frequencies']))
    if args.q:
        partition_functions.append(QuantumHarmonicPF(properties['frequencies']))
    if args.r:
        partition_functions.append(RotationalPF(properties['symmetry'],
                                                properties['moments']))

    # Build superposition approximation using the provided isomer data and the
    # partition functions specified
    if partition_functions:
        superposition = SuperpositionApproximation()
        superposition.add_partition_functions(partition_functions)

        temperature = np.arange(args.min_temp, args.max_temp + 1., 1.)

        results = {'C': superposition.calc_heat_capacity(temperature),
                   'P': superposition.calc_probability(temperature)}
    else:
        print('You must include at least one partition function.')

        results = {}

    # Save the calculated occupation probabilities and heat capacity to a plain
    # text file and plot the results if requested
    for key in results:
        if args.out_file:
            outfile = args.out_file + '_' + key + '.dat'
        else:
            outfile = args.in_file.replace('.xyz', '_') + key + '.dat'

        save_results(results[key], temperature, outfile)

        if args.plot:
            plotfile = outfile.replace('dat', args.plot_format.lower())
            plot_results(results[key], key, temperature, plotfile, args.size)


if __name__ == '__main__':
    main()
