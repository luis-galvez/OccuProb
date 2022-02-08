""" Unit tests for the occuprob.io module """

import numpy as np

from occuprob.io import load_properties_from_extxyz

from occuprob.utils import compare_numpy_dictionaries


def test_load_properties_from_extxyz():
    """ Test loading properties from Extended XYZ files."""

    expected_properties = {'Energies': np.zeros((2,)),
                           'Spin multiplicity': np.ones((2,)),
                           'Frequencies': np.ones((2, 3)),
                           'Moments of inertia': np.array([[0., 2.016, 2.016],
                                                           [1.008, 1.008, 2.016]]),
                           'Symmetry order': np.array([1, 6])}
    loaded_properties = load_properties_from_extxyz('../data/test.xyz')

    assert compare_numpy_dictionaries(expected_properties, loaded_properties)
