"""

The :mod:`flocalx.genetic` module includes an implementation of a genetic
algorithm for optimization of the global explanation theory. It also includes
a chromosome representation of a ruleset for the global explanation theory.
It includes the following classes:

:class:`.Chromosome`
    A representation of a ruleset for the global explanation theory.
:class:`.GeneticAlgorithm`
    A genetic algorithm for optimization of the global explanation theory.
"""

# =============================================================================
# Imports
# =============================================================================

# Local application
from ._chromosome import Chromosome
from ._genetic import GeneticAlgorithm

# =============================================================================
# Public objects
# =============================================================================

__all__ = [
    'Chromosome',
    'GeneticAlgorithm'
]
