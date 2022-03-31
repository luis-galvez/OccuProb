""" Input and output functions """

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

from ase.io import read

from pymatgen.core.structure import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

import matplotlib.pyplot as plt


def calc_symmetry_order(ase_atoms):
    """ Calculates the order of the rotational subgroup of the symmetry point
    group of each structure contained in the input ASE atoms object.

    Parameters
    ----------
    ase_atoms : :obj:`ASE Atoms object`
        Input ASE Atoms object containing the molecule to analyze.

    Returns
    -------
    symmetry_order : float
        Order of the rotational subgroup of the symmetry point group of the
        input molecule.
    """

    symbols = ase_atoms.get_chemical_symbols()
    positions = ase_atoms.get_positions()

    molecule = Molecule(symbols, positions)
    point_group = PointGroupAnalyzer(molecule, eigen_tolerance=0.01)

    symmetry_matrices = np.array([matrix.as_dict()['matrix'] for matrix in
                                  point_group.get_symmetry_operations()])

    symmetry_matrices_det = np.linalg.det(symmetry_matrices)

    symmetry_order = np.count_nonzero(symmetry_matrices_det > 0)

    return symmetry_order


def load_properties_from_extxyz(xyz_filename):
    """ Reads isomer properties (energy, spin_multiplicity, frequencies and
    coordinates) from Extended XYZ files.

    Parameters
    ----------
    xyz_filename : string
        Name of the Extended XYZ filename containing the coordinates of each
        isomer and their properties.

    Returns
    -------
    properties : dict
        Dictionary that maps each key to the correspoding property of each
        isomer in the input file.
    """
    isomers = read(xyz_filename, index=':')

    energies = []
    spin_multiplicity = []
    frequencies = []
    moments_of_inertia = []
    symmetry_order = []

    # Reads the values from the input file
    for atoms in isomers:
        energies.append(atoms.info['Energy'])
        spin_multiplicity.append(atoms.info['Multiplicity'])
        frequencies.append(atoms.info['Frequencies'].flatten(order='F'))
        moments_of_inertia.append(atoms.get_moments_of_inertia())
        symmetry_order.append(calc_symmetry_order(atoms))

    properties = {'energy': np.stack(energies),
                  'multiplicity': np.stack(spin_multiplicity),
                  'frequencies': np.stack(frequencies).astype(np.longdouble),
                  'moments': np.stack(moments_of_inertia),
                  'symmetry': np.stack(symmetry_order)}

    return properties


def plot_results(results, temperature, outfile, **plot_format):
    """ Function to plot the occupation probability, heat capacity or ensemble-
    averaged properties.

    Parameters
    ----------
    results : :obj:`numpy.ndarray`
        A 2D array shape (N, M) containing the data to be plotted.
    temperature : :obj:`numpy.ndarray`
        A 1D array of size M containing the temperature values in K.
    outfile : string
        Output filename.
    plot_format : dict
        Dictionary containing parameters for the figure format.
    """

    labels = plot_format['labels'] if 'labels' in plot_format else None
    ylabel = plot_format['ylabel'] if 'ylabel' in plot_format else None
    ylims = plot_format['ylims'] if 'ylims' in plot_format else None
    size = plot_format['size'] if 'size' in plot_format else (8., 6.)
    linewidth = plot_format['linewidth'] if 'linewidth' in plot_format else 2
    hline_pos = plot_format['hline_pos'] if 'hline_pos' in plot_format else []

    plt.figure(figsize=size)

    plt.xlabel(r'$T$ [K]')
    plt.ylabel(ylabel)

    xmin, xmax = temperature[0], temperature[-1]

    for position in hline_pos:
        plt.hlines(position, xmin, xmax, colors='silver', linestyles='--', lw=2)

    for i, result in enumerate(results):
        if labels:
            plt.plot(temperature, result, label=labels[i], lw=linewidth)
        else:
            plt.plot(temperature, result, lw=linewidth)

    plt.xlim((xmin, xmax))
    plt.ylim(ylims)

    if labels:
        plt.legend()

    plt.tight_layout()
    plt.savefig(outfile)
