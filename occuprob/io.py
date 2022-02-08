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
    properties : dictionary
        Dictionary contaning the properties of each isomer in the input file.
    """
    isomers = read(xyz_filename, index=':')

    energies = []
    spin_multiplicity = []
    frequencies = []
    moments_of_inertia = []
    symmetry_order = []

    # Reads the values from the input file
    for atoms in isomers:
        energies.append(atoms.info['energy'])
        spin_multiplicity.append(atoms.info['multiplicity'])
        frequencies.append(atoms.info['frequencies'].flatten(order='F'))
        moments_of_inertia.append(atoms.get_moments_of_inertia())
        symmetry_order.append(calc_symmetry_order(atoms))

    properties = {'Energies': np.stack(energies),
                  'Spin multiplicity': np.stack(spin_multiplicity),
                  'Frequencies': np.stack(frequencies),
                  'Moments of inertia': np.stack(moments_of_inertia),
                  'Symmetry order': np.stack(symmetry_order)}

    return properties
