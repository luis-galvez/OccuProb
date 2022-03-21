"""A tool for calculating occupation probabilities and ensemble-averaged
properties via the superposition approximation."""

# Add imports here
from .partitionfunctions import *
from .superpositions import *
from .utils import *
from .io import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

from . import _version
__version__ = _version.get_versions()['version']
