# =============================================================================
# Imports
# =============================================================================

# Standard library
from functools import lru_cache

# Third party
import numpy as np
from sklearn.utils import check_X_y
from sklearn.metrics import roc_auc_score, accuracy_score

# Local application
from ..rule import FLocalX

# =============================================================================
# Classes
# =============================================================================

METRIC_FUNCTIONS = {
    'accuracy': accuracy_score,
    'auc': roc_auc_score
}


class GeneticAlgorithm:
    """ Genetic algorithm for optimization of the global explanation theory.
    To execute the genetic algorithm, call the instance of the class.
    """
    def __init__(self,
                 metadata,
                 X,
                 y,
                 kappa=100,
                 crossover_prob=0.8,
                 mutation_prob=0.1,
                 size_pressure=0.4,
                 premise_pressure=0.2,
                 population_size=100,
                 minibatch=None,
                 initial_chromosomes=None,
                 stagnation=False,
                 epsilon=0.001,
                 metric='accuracy',
                 debug=True,
                 random_state=None):
        """ Constructor for the GeneticAlgorithm class.

        Parameters
        ----------
        metadata : dict
            The metadata of the dataset.
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        kappa : int, default=100
            The number of iterations for the genetic algorithm.
        crossover_prob : float, default=0.8
            The probability of crossover.
        mutation_prob : float, default=0.1
            The probability of mutation.
        size_pressure : float, default=0.4
            The pressure for reducing the size of the ruleset.
        premise_pressure : float, default=0.2
            The pressure for reducing the number of premises in the ruleset.
        population_size : int, default=100
            The size of the initial population.
        minibatch : int, default=None
            The size of the minibatch for the genetic algorithm.
            If None, the genetic algorithm will use the entire dataset.
        initial_chromosomes : list of Chromosome, default=None
            The initial chromosomes for the genetic algorithm.
        stagnation : bool, default=False
            Whether to use stagnation as a stopping criterion.
        epsilon : float, default=0.001
            The threshold for stagnation.
        metric : str, default='accuracy'
            The metric to use for the genetic algorithm. It can be 'accuracy' or 'auc'.
        debug : bool, default=True
            Whether to print debug information.
        random_state : int, RandomState instance or None, default=None
            Determines random number generation for the genetic algorithm.
        """

        if random_state is None:
            self.random_state = np.random.default_rng()
        else:
            self.random_state = random_state

        self.metadata = metadata
        self.X, self.y = check_X_y(X, y, dtype=['float64', 'object'])
        self.kappa = kappa
        self.current_iteration = 0
        self.total_iterations = 0
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.debug = debug
        self.size_pressure = size_pressure
        self.premise_pressure = premise_pressure
        if (size_pressure + premise_pressure) > 1:
            raise ValueError('The sum of size_pressure and premise_pressure must be smaller than 1')

        self.population_size = population_size
        self.minibatch = minibatch
        self.stagnation = stagnation
        self.epsilon = epsilon
        self.metric = metric
        self.population = self._initialize_population(initial_chromosomes)
        self.best_score = np.max(self.population)

        # Multiplier constant for rank selection
        RANK_MULTIPLIER = 2/(len(self.population)**2 + (len(self.population)))
        # Array of rank probabilities
        self.rank_probability = [(len(self.population) - (i+1) + 1) * RANK_MULTIPLIER
                                 for i in range(len(self.population))]

    def __call__(self):
        self.current_iteration = 0
        self.total_iterations = 0
        while not self._finish():
            self.current_iteration += 1
            self.total_iterations += 1
            if self.debug:
                print(f'Iteration {self.current_iteration}, max score: {np.max(self.population)}')
            population = self._rank_selection()
            population = self._crossover(population)
            population = self._mutation(population)
            self.population = self._elitism(population)

    def _finish(self):
        if self.stagnation and np.abs(self.best_score - np.max(self.population)) >= self.epsilon:
            self.best_score = np.max(self.population)
            self.current_iteration = 0
            return False
        return self.current_iteration >= self.kappa

    def _initialize_population(self, initial_chromosomes):
        initial_population = []
        if initial_chromosomes:
            for chromosome in initial_chromosomes:
                chromosome.fitness = self.fitness
                n_chromosomes = self.population_size // len(initial_chromosomes)
                initial_population += chromosome.generate_initial_population(self.metadata,
                                                                             n_chromosomes)
            return initial_population
        else:
            return np.zeros(self.population_size, dtype=object)

    def _selection(self):
        if self.debug:
            print('Selection')
        p = np.array([chromosome.score for chromosome in self.population])
        p = p / p.sum()
        return self.random_state.choice(self.population, size=len(self.population), p=p)

    def _rank_selection(self):
        if self.debug:
            print('Rank selection')
        sorted_population = np.sort(self.population)[::-1]
        return self.random_state.choice(sorted_population, size=len(self.population), p=self.rank_probability)

    def _crossover(self, population):
        if self.debug:
            print('Crossover')
            counter = 0
        new_population = np.copy(population)
        for i in range(0, len(new_population), 2):
            child1, child2 = new_population[i], new_population[i+1]
            if self.random_state.random() < self.crossover_prob:
                # if self.debug:
                #     pass
                #     print(f'Crossing pair {counter} and {counter + 1}')
                child1, child2 = child1.crossover(child2)
            if self.debug:
                counter += 2
            new_population[i], new_population[i+1] = child1, child2
        return new_population

    def _mutation(self, population):
        if self.debug:
            print('Mutation')
        new_population = np.copy(population)
        for i in range(len(new_population)):
            if self.random_state.random() < self.mutation_prob:
                new_population[i] = new_population[i].mutation(self.metadata)
        return new_population

    def _elitism(self, population):
        if self.debug:
            print('Elitism')
        new_population = np.sort(population)[::-1]
        new_population[-1] = np.max(self.population)
        if self.debug:
            print(f'Best chromosome: {np.max(new_population)}')
        return new_population

    @lru_cache(maxsize=10000)
    def _antecedent_match(self, antecedent, value):
        return antecedent.match(value)

    @lru_cache(maxsize=10000)
    def _rule_match(self, rule_antecedent, x):
        try:
            match = np.min([self._antecedent_match(antecedent, x[antecedent.variable])
                            for antecedent in rule_antecedent])
        except Exception:
            match = 0
        return match

    def fitness(self, chromosome):
        rb = FLocalX.from_chromosome(chromosome, self.metadata)
        n_premises = rb.rule_size()
        size_fitness = self.size_pressure * (1 - np.sum(chromosome.used_rules) / len(chromosome.used_rules))
        premise_fitness = self.premise_pressure * (1 - n_premises / len(self.metadata))
        score_fitness = (1 - self.size_pressure - self.premise_pressure) * self.cache_score(rb)

        return size_fitness + premise_fitness + score_fitness

    def _minibatch(self, X, y):
        val, amount = np.unique(y, return_counts=True)
        idx = []
        for v, a in zip(val, amount):
            id = np.random.choice(np.where(y == v)[0], int(a/len(y) * self.minibatch), replace=True)
            idx += list(id)

        return X[idx], y[idx]

    def cache_score(self, rb):
        rules = rb.rules
        random_guesses = 0
        if not rb.rules:
            return 0
        if self.minibatch:
            X, y = self._minibatch(self.X, self.y)
        else:
            X, y = self.X, self.y
        # Aggregated vote
        # First, we get the sum of the match values for each class
        predictions = np.zeros(len(X))
        for i, x in enumerate(X):
            votes = {}
            tx = tuple(x)
            for rule in rules:
                if rule.consequent not in votes:
                    votes[rule.consequent] = 0

                votes[rule.consequent] += self._rule_match(rule.antecedent, tx)

            # Then, we get the class with the highest sum
            if votes and max(votes.values()) > 0:
                predictions[i] = max(votes, key=votes.get)
            else:
                predictions[i] = self.random_state.integers(0, 2)  # Random guess if no rules match
                random_guesses += 1

        return METRIC_FUNCTIONS[self.metric](y, predictions) * (1 - random_guesses / len(X))
