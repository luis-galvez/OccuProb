""" Superposition approximations to the PES """

import numpy as np

import occuprob.partitionfunctions as pf


class SuperpositionApproximation():
    """
    Represents a superposition approximation of the PES.

    ...

    Methods
    -------
    calc_partition_functions(temperature):
        Calculates the individual partition function contributions of each
        geometrically unique isomer
    calc_probability(temperature):
        Calculates the occupation probability in the temperature range provided.
    calc_ensemble_average(temperature, observable):
        Calculates the ensemble average for the given observable in the provided
        temperature range.
    calc_heat_capacity(temperature):
        Calculates the canonical heat capacity for the given temperature range.
    """

    def __init__(self, potential_energy, frequencies, symmetry_order,
                 spin_multiplicity):
        self.potential_energy = potential_energy
        self.frequencies = frequencies
        self.symmetry_order = symmetry_order
        self.spin_multiplicity = spin_multiplicity

        self.global_minimum = np.argmin(self.potential_energy)
        self.partition_functions = []

    def calc_partition_functions(self, temperature):
        """ Calculates the individual partition functions of each geometrically
        unique isomer.

        Parameters
        ----------
        temperature : array_like
            Temperature range in K.

        Returns
        -------
        partition_functions : array_like
            Individual partition function contributions for each isomer.
        """
        if not self.partition_functions:
            print("You must include at least one partition function.")
            return None

        contributions = [partition_function.calc_part_func(temperature) for
                         partition_function in self.partition_functions]
        partition_functions = np.prod(np.stack(contributions), axis=0)

        return partition_functions

    def calc_probability(self, temperature):
        """
        Calculates the occupation probability in the temperature range provided.

        Parameters
        ----------
        temperature : array_like
            Temperature range in K.

        Returns
        -------
        occupation_probability : array_like
            Occupation probability.
        """
        # Calculates the individual partition function contributions
        partition_functions = self.calc_partition_functions(temperature)

        # The total partition function is the sum of all the individual
        # contributions of each geometrically unique isomer
        total_partition_function = np.sum(partition_functions, axis=0)

        # The probability of finding the system in some isomer is calculated
        # as the quotient of its individual partition function divided by the
        # total partition function
        occupation_probability = partition_functions / total_partition_function

        return occupation_probability

    def calc_ensemble_average(self, temperature, observable):
        """
        Calculates the ensemble average for the given observable in the provided
        temperature range.

        Parameters
        ----------
        temperature : array_like
            Temperature range in eV.
        observable : array_like
            Input observable.

        Returns
        -------
        calc_ensemble_average : array_like
            Occupation probability.
        """

        # The ensemble average of a given observable is calculated as a
        # weighted sum using the occupation probabilities of each unique isomer
        # as the weights
        probability = self.calc_probability(temperature)
        ensemble_average = np.sum(observable[:, None] * probability, axis=0)

        return ensemble_average

    def calc_heat_capacity(self, temperature):
        """
        Calculates the canonical heat capacity for the given temperature range.

        Parameters
        ----------
        temperature : array_like
            Temperature range in eV.

        Returns
        -------
        heat_capacity : array_like
            Canonical heat capacity.
        """

        heat_capacity = np.exp(self.calc_probability(temperature))

        return heat_capacity


class ClassicalHarmonicSA(SuperpositionApproximation):
    """
    Represents a classical harmonic superposition approximation of the PES.

    ...

    Attributes
    ----------
    potential_energy : array_like
        Potential energy values (in eV) of each geometrically unique isomer.
    frequencies : array_like
        Frequency values corresponding to each geometrically unique isomer.
    symmetry_order : array_like
        Order of the point group symmetry of each geometrically unique isomer.

    Methods
    -------
    calc_probability(temperature):
        Calculates the occupation probability in the temperature range provided.
    calc_heat_capacity(temperature):
        Calculates the canonical heat capacity for the given temperature range.
    calc_average_observable(temperature, observable):
        Calculates the ensemble average for the given observable in the provided
        temperature range.
    """
    def __init__(self, potential_energy, frequencies, symmetry_order):
        super().__init__(potential_energy, frequencies, symmetry_order,
                         np.ones_like(potential_energy))

        # Partition functions
        self.partition_functions = [pf.ElectronicPF(self.potential_energy,
                                                    self.symmetry_order,
                                                    self.spin_multiplicity),
                                    pf.ClassicalHarmonicPF(self.frequencies)]


class QuantumHarmonicSA(SuperpositionApproximation):
    """
    Represents a classical harmonic superposition approximation of the PES.

    ...

    Attributes
    ----------
    potential_energy : array_like
        Potential energy values (in eV) of each geometrically unique isomer.
    frequencies : array_like
        Frequency values corresponding to each geometrically unique isomer.
    symmetry_order : array_like
        Order of the point group symmetry of each geometrically unique isomer.

    Methods
    -------
    calc_probability(temperature):
        Calculates the occupation probability in the temperature range provided.
    calc_heat_capacity(temperature):
        Calculates the canonical heat capacity for the given temperature range.
    calc_average_observable(temperature, observable):
        Calculates the ensemble average for the given observable in the provided
        temperature range.
    """
    def __init__(self, potential_energy, frequencies, symmetry_order,
                 spin_multiplicity):
        super().__init__(potential_energy, frequencies, symmetry_order,
                         spin_multiplicity)

        # Partition functions
        self.partition_functions = [pf.ElectronicPF(self.potential_energy,
                                                    self.symmetry_order,
                                                    self.spin_multiplicity),
                                    pf.QuantumHarmonicPF(self.frequencies)]
