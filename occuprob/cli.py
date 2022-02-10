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


def main():
    """ Command-line interface """
    parser = argparse.ArgumentParser(description='OccuProb.')
    electronic = parser.add_mutually_exclusive_group()
    electronic.add_argument('-e', '-E', action='store_true',
                            help='Electronic partition function')
    electronic.add_argument('-s', '-S', action='store_true',
                            help='Electronic partition function including spin')
    vibrational = parser.add_mutually_exclusive_group()
    vibrational.add_argument('-v', '-V', action='store_true',
                             help='Classical harmonic vibrational partition function')
    vibrational.add_argument('-q', '-Q', action='store_true',
                             help='Quantum harmonic vibrational partition function')
    parser.add_argument('-r', '-R', action='store_true',
                        help='Rotational partition function')
    parser.add_argument('input_file', help='Extended XYZ file contaning the list of isomers')
    parser.add_argument('--output', help='Output filename prefix')
    parser.add_argument('--min_temp', type=float, default=0.,
                        help='Maximum temperature in K (default: 0)')
    parser.add_argument('--max_temp', type=float, default=500.,
                        help='Maximum temperature in K (default: 500)')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the results and save them as image files')
    parser.add_argument('--size', type=float, nargs=2, default=[8., 6.],
                        help='Width and height of the output image, in inches (default: 8.0 6.0)')
    args = parser.parse_args()

    properties = io.load_properties_from_extxyz(args.input_file)

    partition_functions = []

    if args.e:
        partition_functions.append(ElectronicPF(properties['energy'],
                                                np.ones_like(properties['energy'])))
    if args.s:
        partition_functions.append(ElectronicPF(properties['energy'],
                                                properties['multiplicity']))
    if args.v:
        partition_functions.append(ClassicalHarmonicPF(properties['frequencies']))
    if args.q:
        partition_functions.append(QuantumHarmonicPF(properties['frequencies']))
    if args.r:
        partition_functions.append(RotationalPF(properties['symmetry'],
                                                properties['moments']))

    if partition_functions:
        superposition = SuperpositionApproximation(partition_functions)

        temperature = np.arange(args.min_temp, args.max_temp + 1., 1.)

        results = {'p': superposition.calc_probability(temperature),
                   'c': superposition.calc_heat_capacity(temperature)}

        for key in results:
            if args.output:
                outfile = args.output + '_' + key
            else:
                outfile = args.input_file.replace('.xyz', '_' + key)

            if args.plot:
                io.plot_results(results[key], temperature, outfile + '.pdf',
                                args.size, result_type=key)

            outdata = np.vstack((temperature, results[key]))
            np.savetxt(outfile + '.dat', outdata.T)

    else:
        print('You must include at least one partition function.')


if __name__ == '__main__':
    main()
