#!/usr/bin/env python
# -*- coding: utf -*-

from __future__ import absolute_import, print_function, division


import io
import itertools
import random
import sys


TOP_WEIGHT = 0


class WCNFException(Exception):
    """Invalid MaxSAT operation."""


class WCNFFormula(object):
    """Simple WCNF formula class."""

    def __init__(self):
        self.num_vars = 0
        self.hard = []    # type: List[List[int]]
        self.soft = []    # type: List[Tuple[weight, List[int]]]
        self._sum_soft_weights = 0
        self.header = []  # type: List[str]

    @property
    def num_clauses(self):
        """num_clauses() -> int

        Number of clauses in this formula.
        """
        return len(self.hard) + len(self.soft)

    @property
    def top_weight(self):
        """top_weight() -> int

        Sum of all the soft weights + 1.
        """
        return self._sum_soft_weights + 1

    def clean(self):
        """Re-initializes the wcnf formula instance."""
        self.__init__()

    def add_clauses(self, clauses, weight=TOP_WEIGHT):
        """Adds the given set of clauses, having each one the specified weight.

        :param clauses: Iterable filled with sets of literals.
        :type clauses: list[list[int]]
        :param weight: Weight applied to all the clauses, as in add_clause().
        :type weight: int
        """
        for c in clauses:
            self.add_clause(c, weight)

    def add_clause(self, literals, weight):
        """Adds the given literals as a new clause with the specified weight.

        :param literals: Clause literals
        :type literals: list[int]
        :param weight: Clause weight, less than 1 means infinity.
        :type weight: int
        """
        self._check_literals(literals)
        self._add_clause(literals, weight)

    def add_exactly_one(self, literals, weight):
        """Adds the necessary combination of clauses to ensure that exactly
        one of the given literals evaluates to true.

        :param literals: Literals to include in the exactly one set of clauses.
        :type literals: list[int]
        :param weight: Clauses weight, less than 1 means infinity.
        :type weight: int
        """
        self._check_literals(literals)
        self._add_at_least_one(literals, weight)
        self._add_at_most_one(literals, weight)

    def add_at_least_one(self, literals, weight):
        """Adds the necessary combination of clauses to ensure that at least
        one of the given literals evaluates to true.

        :param literals: Literals to include in the at most one set of clauses.
        :type literals: list[int]
        :param weight: Clause weight, less than 1 means infinity.
        :type weight: int
        """
        self._check_literals(literals)
        self._add_at_least_one(literals, weight)

    def add_at_most_one(self, literals, weight):
        """Adds the necessary combination of clauses to ensure that at most
        one of the given literals evaluates to true.

        :param literals: Literals to include in the at most one set of clauses.
        :type literals: list[int]
        :param weight: Clauses weight, less than 1 means infinity.
        :type weight: int
        """
        self._check_literals(literals)
        self._add_at_most_one(literals, weight)

    def new_var(self):
        """Returns the next free variable of this formula.

        :return: The next free variable (>1).
        :rtype: int
        """
        self.num_vars += 1
        return self.num_vars

    def extend_vars(self, how_many):
        """Extends the number of used variables.
        """
        if how_many < 0:
            raise ValueError("Cannot be extended a negative quantity")
        self.num_vars += how_many

    def write_dimacs(self, stream=sys.stdout):
        """Writes the formula in DIMACS format into the specified stream.

        :param stream: A writable stream object.
        """
        for line in self.header:
            print("c", line, file=stream)

        tw = self.top_weight
        print("p wcnf %d %d %d" % (self.num_vars, self.num_clauses, tw),
              file=stream)

        print("c ===== Hard Clauses =====", file=stream)
        for c in self.hard:
            print(tw, " ".join(str(l) for l in c), "0", file=stream)

        print("c ===== Soft Clauses (Sum weights: {0}) ====="
              .format(self._sum_soft_weights), file=stream)
        for w, c in self.soft:
            print(w, " ".join(str(l) for l in c), "0", file=stream)

    def write_dimacs_file(self, file_path):
        """Writes the formula in DIMACS format into the specified file.

        :param file_path: Path to a writable file.
        :type file_path: str
        """
        with open(file_path, 'w') as f:
            self.write_dimacs(f)

    def _add_clause(self, literals, weight):
        if weight < 1:
            self.hard.append(literals)
        else:
            self.soft.append((weight, literals))
            self._sum_soft_weights += weight

    def _add_at_least_one(self, literals, weight):
        # **** Your code here ****
        raise NotImplementedError()

    def _add_at_most_one(self, literals, weight):
        # **** Your code here ****
        raise NotImplementedError()

    def _check_literals(self, literals):
        for var in map(abs, literals):
            if var == 0:
                raise WCNFException("Clause cannot contain variable 0")
            elif self.num_vars < var:
                raise WCNFException("Clause contains variable {0}, not defined"
                                    " by new_var()".format(var))

    def __str__(self):
        s_io = io.StringIO()
        self.write_dimacs(stream=s_io)
        output = s_io.getvalue()
        s_io.close()
        return output


# Utilitie functions
###############################################################################

def generate_3sat_gadget(formula, clause):
    """generate_3sat_gadget(formula, clause) -> List[List[int]]

    Generates the 3SAT equivalent of the given clause.
    New auxiliar variables are created in the given formula.

    :return: 3SAT equivalent clauses of the given one.
    """
    if not clause:
        raise ValueError("An empty clause cannot be transformed to 3SAT")

    # **** Your code here ****
    raise NotImplementedError()

    return clauses


def formula_to_1_3_wpm(formula):
    """Transforms this formula to its 1,3 WPM equivalent.

    :return: A new formula whose clauses are the 1,3 WPM
             equivalent of the input formula.
    """
    new_f = WCNFFormula()
    new_f.header = list(formula.header)
    new_f.header.append(" **** 1,3-WPM transformed formula ****")
    new_f.header.append("")

    # **** Your code here ****
    raise NotImplementedError()

    return new_f
