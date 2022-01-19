"""Tool to calculate thermodynamic properties using the superposition approximation of the PES."""

# Add imports here
from .occuprob import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
