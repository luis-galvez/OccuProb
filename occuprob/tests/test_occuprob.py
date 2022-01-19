"""
Unit and regression test for the occuprob package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import occuprob


def test_occuprob_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "occuprob" in sys.modules
