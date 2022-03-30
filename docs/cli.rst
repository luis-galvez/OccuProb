Command-line interface
======================

.. autosummary::
   :toctree: autosummary

OccuProb includes a basic command-line interface (CLI) to calculate the occupation
probability and heat capacity for a set of isomers in Extended-XYZ format files.

Input files format
------------------

Extended XYZ.

Usage
-----
.. code-block:: console

	occuprob -sqr Pt5_isomers.xyz --plot --max_temp=1000 --size 6 4
