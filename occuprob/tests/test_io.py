""" Unit tests for the occuprob.io module """

import os

import numpy as np

import occuprob
from occuprob.io import load_properties_from_extxyz
from occuprob.utils import compare_numpy_dictionaries


def test_load_properties_from_extxyz():
    """ Test loading properties from Extended XYZ files."""

    expected_properties = {'Energies': np.zeros((2,)),
                           'Spin multiplicity': 2. * np.ones((2,)),
                           'Frequencies': np.ones((2, 3)),
                           'Moments of inertia': np.array([[0., 2.016, 2.016],
                                                           [1.008, 1.008, 2.016]]),
                           'Symmetry order': np.array([1, 6])}

    test_file = os.path.dirname(occuprob.__file__) + '/data/test.xyz'
    loaded_properties = load_properties_from_extxyz(test_file)

    assert compare_numpy_dictionaries(expected_properties, loaded_properties)
