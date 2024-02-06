""" 

The :mod:`flocalx.rule` module includes the necessary elements to build a rule-based
representation for the local explanation theory. These elements can be put together
into three groups: 

1. Antecedents: 
    - :class:`.NumericAntecedent`: This class represents a numeric antecedent of a rule.
    - :class:`.CategoricalAntecedent`: This class represents a categorical antecedent of a rule.
    - :class:`.FuzzyAntecedent`: This class represents a fuzzy antecedent of a rule.

2. Rules:
    - :class:`.Rule`: This class represents a crisp rule.
    - :class:`.FuzzyRule`: This class represents a fuzzy rule.

3. Rulesets:
    - :class:`.RuleSet`: This is the base class that represents a crisp ruleset and
      from which the rest of the rulesets inherit.
    - :class:`.FuzzyRuleSet`: This class that represents a fuzzy ruleset.
    - :class:`.FLocalX`: This class represents a FLocalX explanation theory, which is a 
      fuzzy ruleset that can be optimized using a genetic algorithm from the module 
      :mod:`flocalx.genetic`.

"""

# =============================================================================
# Imports
# =============================================================================

# Local application
from ._rule import Rule, FuzzyRule, NumericAntecedent, CategoricalAntecedent, FuzzyAntecedent, HelloWorldRule
from ._ruleset import RuleSet, FuzzyRuleSet, FLocalX

# =============================================================================
# Public objects
# =============================================================================

__all__ = [
    "NumericAntecedent",
    "CategoricalAntecedent",
    "FuzzyAntecedent",
    "Rule",
    "FuzzyRule",
    "RuleSet",
    "FuzzyRuleSet",
    "FLocalX"
    "HelloWorldRule",
]
