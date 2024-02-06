# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
from ..utils import get_fuzzy_variables


# =============================================================================
# Classes
# =============================================================================


class Chromosome:
    """ A representation of a ruleset for the global explanation theory used
    in the genetic algorithm.
    """
    def __init__(self, variables, rules, modifiers, used_rules, alpha=0.8, fitness=lambda x: 0, random_state=None):
        """ Constructor for the Chromosome class.

        Parameters
        ----------
        variables : numpy.ndarray
            The variables of the chromosome.
        rules : numpy.ndarray
            The rules of the chromosome.
        modifiers : numpy.ndarray
            The modifiers of the chromosome.
        used_rules : numpy.ndarray
            The used rules of the chromosome.
        alpha : float, optional
            The alpha parameter for the crossover operation, by default 0.8.
        fitness : function, optional
            The fitness function for the chromosome, by default lambda x: 0.
        random_state : numpy.random.Generator, optional
            The random state for the chromosome, by default None.
        """
        self.variables = variables
        self.rules = rules
        self.modifiers = modifiers
        self.used_rules = used_rules
        self.genes = np.concatenate([variables.flatten(), rules.flatten(), modifiers.flatten(), used_rules.flatten()])
        self.alpha = alpha

        # Fitness es una funcion que toma como parametro el cromosoma y devuelve un valor numerico
        self.fitness = fitness
        self.score = fitness(self)

        if random_state is None:
            self.random_state = np.random.default_rng()
        else:
            self.random_state = random_state

    def __repr__(self) -> str:
        return f"Chromosome({self.genes}, {self.score})"

    def __hash__(self) -> int:
        return hash((tuple(self.genes), self.score))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Chromosome):
            return False
        return hash(self) == hash(o)

    def __lt__(self, o: object) -> bool:
        return self.score < o.score

    def __gt__(self, o: object) -> bool:
        return self.score > o.score

    def __le__(self, o: object) -> bool:
        return self.score <= o.score

    def __ge__(self, o: object) -> bool:
        return self.score >= o.score

    def __sub__(self, other):
        return self.score - other.score

    @staticmethod
    def _rules_chromosome(ruleset, variable_metadata):
        return np.array([rule.chromosome(variable_metadata) for rule in ruleset.rules])

    @staticmethod
    def _rules_modifier_chromosome(ruleset, variable_metadata):
        return np.array([rule.modifier_chromosome(variable_metadata) for rule in ruleset.rules])

    @staticmethod
    def _variables_chromosome(ruleset, metadata):
        c = np.zeros((len(metadata['continuous']), metadata['sets'] - 2))
        for var in metadata['continuous']:
            c[metadata['continuous'][var], :] = np.array(list(metadata[var]['points'].keys())[1:-1])
        return c

    @staticmethod
    def from_ruleset(ruleset, variable_metadata, alpha=0.8, fitness=lambda x: 0):
        """ Create a chromosome from a ruleset by mapping the ruleset to the
        chromosome representation according to the variable metadata.

        Parameters
        ----------
        ruleset : Ruleset
            The ruleset to be mapped to the chromosome.
        variable_metadata : dict
            The metadata of the variables.
        alpha : float, optional
            The alpha parameter for the crossover operation, by default 0.8.
        fitness : function, optional
            The fitness function for the chromosome, by default lambda x: 0.

        Returns
        -------
        Chromosome
            A chromosome representation of the ruleset.
        """
        variables = Chromosome._variables_chromosome(ruleset, variable_metadata)
        rules = Chromosome._rules_chromosome(ruleset, variable_metadata)
        modifiers = Chromosome._rules_modifier_chromosome(ruleset, variable_metadata)
        used_rules = np.ones(len(ruleset.rules))

        return Chromosome(variables, rules, modifiers, used_rules, alpha, fitness, ruleset.random_state)

    def _fuzzy_variables(self, metadata):
        fuzzy_points = {}
        for order in metadata['continuous']:
            fuzzy_points[metadata[order]['name']] = [metadata[order]['min']] \
                                                    + sorted(self.variables[metadata['continuous'][order]]) \
                                                    + [metadata[order]['max']]

        return get_fuzzy_variables(fuzzy_points, metadata['discrete_fuzzy_values'], metadata['fuzzy_variables_order'])

    def _max_min_arithmetic_crossover(self, parent1, parent2):
        child1 = np.min([parent1, parent2], axis=0)
        child2 = np.max([parent1, parent2], axis=0)
        child3 = self.alpha * parent1 + (1 - self.alpha) * parent2
        child4 = self.alpha * parent2 + (1 - self.alpha) * parent1
        return child1, child2, child3, child4

    def _two_point_crossover(self, parent1, parent2):
        a, b = sorted(self.random_state.integers(0, len(parent1), 2))

        child1 = np.concatenate([parent1[:a], parent2[a:b], parent1[b:]])
        child2 = np.concatenate([parent2[:a], parent1[a:b], parent2[b:]])

        return child1, child2

    def crossover(self, other):
        """Crossover operation for the chromosome.

        Parameters
        ----------
        other : Chromosome
            The other chromosome to be crossed over with.

        Returns
        -------
        list
            A list of the two best children chromosomes.
        """

        variables_children = self._max_min_arithmetic_crossover(self.variables.flatten(),
                                                                other.variables.flatten())
        rules_child_1, rules_child_2 = self._two_point_crossover(self.rules.flatten(),
                                                                 other.rules.flatten())
        modifiers_child_1, modifiers_child_2 = self._two_point_crossover(self.modifiers.flatten(),
                                                                         other.modifiers.flatten())
        used_rules_child_1, used_rules_child_2 = self._two_point_crossover(self.used_rules.flatten(),
                                                                           other.used_rules.flatten())

        children = []
        for var_child in np.unique(np.array(variables_children).round(3), axis=0):
            child_1 = Chromosome(var_child.reshape(self.variables.shape),
                                 rules_child_1.reshape(self.rules.shape),
                                 modifiers_child_1.reshape(self.modifiers.shape),
                                 used_rules_child_1.reshape(self.used_rules.shape),
                                 self.alpha,
                                 self.fitness,
                                 self.random_state)

            child_2 = Chromosome(var_child.reshape(self.variables.shape),
                                 rules_child_2.reshape(self.rules.shape),
                                 modifiers_child_2.reshape(self.modifiers.shape),
                                 used_rules_child_2.reshape(self.used_rules.shape),
                                 self.alpha,
                                 self.fitness,
                                 self.random_state)

            children.append(child_1)
            children.append(child_2)

        return sorted(children, reverse=True)[:2]

    def _variables_mutation(self, metadata):
        new_variables = np.copy(self.variables)
        m, n = self.random_state.integers(self.variables.shape[0]), self.random_state.integers(self.variables.shape[1])
        v = list(metadata['continuous'].keys())[m]
        min_val, max_val = metadata[v]['min'], metadata[v]['max']
        new_val = self.random_state.uniform(min_val, max_val)
        new_variables[m, n] = new_val
        return new_variables

    def _rules_mutation(self, metadata):
        new_rules = np.copy(self.rules)
        m, n = self.random_state.integers(self.rules.shape[0]), self.random_state.integers(self.rules.shape[1]-2)
        if 'values' in metadata[n]:
            length = len(metadata[n]['values'])
        else:
            length = len(metadata[n]['points'])
        new_rules[m, n] = self.random_state.integers(-1, length)
        return new_rules

    def _modifiers_mutation(self):
        new_modifiers = np.copy(self.modifiers)
        m, n = self.random_state.integers(self.modifiers.shape[0]), self.random_state.integers(self.modifiers.shape[1])
        new_modifiers[m, n] = self.random_state.integers(-1, 2)
        return new_modifiers

    def _used_rules_mutation(self):
        new_used_rules = np.copy(self.used_rules)
        m = self.random_state.integers(self.used_rules.shape[0])
        new_used_rules[m] = 1 - new_used_rules[m]
        return new_used_rules

    def mutation(self, metadata):
        """ Mutation operation for the chromosome.

        Returns
        -------
        Chromosome
            A new chromosome with the mutation applied.

        """
        new_variables = self._variables_mutation(metadata)
        new_rules = self._rules_mutation(metadata)
        new_modifiers = self._modifiers_mutation()
        new_used_rules = self._used_rules_mutation()
        return Chromosome(new_variables,
                          new_rules,
                          new_modifiers,
                          new_used_rules,
                          self.alpha,
                          self.fitness,
                          self.random_state)

    def generate_initial_population(self, metadata, population_size):
        """ Generate an initial population for the genetic algorithm.

        Parameters
        ----------
        metadata : dict
            The metadata of the variables.
        population_size : int
            The size of the population to be generated.

        Returns
        -------
        list
            A list of the initial population.
        """
        population = []
        pop_size = population_size // 4
        for i in range(pop_size):
            new_variables = self.random_variables(metadata)
            new_chromosome = Chromosome(new_variables,
                                        self.rules,
                                        self.modifiers,
                                        self.used_rules,
                                        self.alpha,
                                        self.fitness,
                                        self.random_state)
            population.append(new_chromosome)

        for i in range(pop_size):
            new_rules = self.random_rules(metadata)
            new_chromosome = Chromosome(self.variables,
                                        new_rules,
                                        self.modifiers,
                                        self.used_rules,
                                        self.alpha,
                                        self.fitness,
                                        self.random_state)
            population.append(new_chromosome)

        for i in range(pop_size):
            new_modifiers = self.random_modifiers()
            new_chromosome = Chromosome(self.variables,
                                        self.rules,
                                        new_modifiers,
                                        self.used_rules,
                                        self.alpha,
                                        self.fitness,
                                        self.random_state)
            population.append(new_chromosome)

        for i in range(pop_size):
            new_used_rules = self.random_used_rules()
            new_chromosome = Chromosome(self.variables,
                                        self.rules,
                                        self.modifiers,
                                        new_used_rules,
                                        self.alpha,
                                        self.fitness,
                                        self.random_state)
            population.append(new_chromosome)

        return population

    def random_variables(self, metadata):
        new_variables = np.copy(self.variables)
        for i in range(self.variables.shape[0]):
            for j in range(self.variables.shape[1]):
                v = list(metadata['continuous'].keys())[i]
                min_val, max_val = metadata[v]['min'], metadata[v]['max']
                new_variables[i, j] = self.random_state.uniform(min_val, max_val)
        return new_variables

    def random_rules(self, metadata):
        new_rules = np.copy(self.rules)
        for i in range(self.rules.shape[0]):
            for j in range(self.rules.shape[1]-2):
                if 'values' in metadata[j]:
                    length = len(metadata[j]['values'])
                else:
                    length = len(metadata[j]['points'])
                new_rules[i, j] = self.random_state.integers(-1, length)
        return new_rules

    def random_modifiers(self):
        new_modifiers = np.copy(self.modifiers)
        for i in range(self.modifiers.shape[0]):
            for j in range(self.modifiers.shape[1]):
                new_modifiers[i, j] = self.random_state.integers(-1, 2)
        return new_modifiers

    def random_used_rules(self):
        new_used_rules = np.copy(self.used_rules)
        for i in range(self.used_rules.shape[0]):
            new_used_rules[i] = self.random_state.integers(0, 2)
        return new_used_rules
