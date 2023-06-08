from sklearn.utils import check_X_y
from sklearn.metrics import accuracy_score
import numpy as np

class GeneticAlgorithm:
    def __init__(self, metadata, X, y, iterations=100, crossover_prob=0.8, mutation_prob=0.1, size_pressure=0.5, population_size=100, initial_ruleset=None, debug=True):
        self.rule_cache = {}
        self.variable_cache = {}
        self.metadata = metadata
        self.X, self.y = check_X_y(X, y, dtype=['float64', 'object'])
        self.iterations = range(iterations)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.debug = debug
        self.size_pressure = size_pressure
        self.population_size = population_size
        RANK_MULTIPLIER = 2/(population_size**2 + (population_size)) # Multiplier constant for rank selection
        self.rank_probability = [(population_size - (i+1) + 1) * RANK_MULTIPLIER for i in range(population_size)] # Probability of selection for each chromosome for rank selection
        self.population = self._initialize_population(initial_ruleset)


    def __call__(self):
        for i in self.iterations:
            if self.debug:
                print(f'Iteration {i}')
            population = self._selection()
            population = self._crossover(population)
            population = self._mutation(population)
            self.population = self._elitism(population)

    def _initialize_population(self, initial_ruleset):
        if initial_ruleset:
            initial_chromosome = initial_ruleset.chromosome(variable_metadata=self.metadata, alpha=0.7, fitness=self.fitness)
            return initial_chromosome.generate_initial_population(self.metadata, self.population_size)
        else:
            return np.zeros(self.population_size, dtype=object)
        
    
    def _selection(self):
        if self.debug:
            print('Selection')
        p = np.array([chromosome.score for chromosome in self.population])
        p = p / p.sum()
        return np.random.choice(self.population, size=len(self.population), p=p)
    
    def _rank_selection(self):
        if self.debug:
            print('Rank selection')
        sorted_population = np.sort(self.population)[::-1]
        return np.random.choice(sorted_population, size=len(self.population), p=self.rank_probability)

    def _crossover(self, population):
        if self.debug:
            print('Crossover')
            counter = 0
        new_population = np.copy(population)
        for i in range(0, len(new_population), 2):
            child1, child2 = new_population[i], new_population[i+1]
            if np.random.random() < self.crossover_prob:
                if self.debug:
                    print(f'Crossing pair {counter} and {counter + 1}')
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
            if np.random.random() < self.mutation_prob:
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

    def _antecedent_match(self, antecedent, value):
        if antecedent.variable in self.metadata['continuous']:
            key = tuple((antecedent, value))
            if key in self.variable_cache:
                # print(f'Hit cache for {antecedent} and {value}')
                return self.variable_cache[key]
            else:
                match = antecedent.match(value)
                self.variable_cache[key] = match
        else:
            match = antecedent.match(value)
        return match

    def _rule_match(self, rule, x, rule_chromosome_rep, mod_chromosome_rep, vars_chromosome_rep):
        key = tuple((x.tobytes(), rule_chromosome_rep.tobytes(), mod_chromosome_rep.tobytes(), vars_chromosome_rep.tobytes()))
        # if False:
        if key in self.rule_cache:
            # print(f'Hit cache for {rule} and {x}')
            return self.rule_cache[key]
        else:
            match = min([self._antecedent_match(antecedent, x[antecedent.variable]) for antecedent in rule.antecedent])
            self.rule_cache[key] = match
            return match
        
    def fitness(self, chromosome):
        return self.size_pressure * (1 - np.sum(chromosome.used_rules) / len(chromosome.used_rules)) + (1 - self.size_pressure) * self.score(chromosome)
        
    def score(self, chromosome):
        rules = chromosome.to_rule_based_system(self.metadata).rules

        ## Aggregated vote
        ## First, we get the sum of the match values for each class
        predictions = []
        votes = {}
        for x in self.X:
            for i, rule in enumerate(rules):
                if rule.consequent not in votes:
                    votes[rule.consequent] = 0
                
                # TODO: FIX CACHE, ONLY SELECT THE VARIABLES THAT APPEAR IN THE RULE, NOW IT'S ALL OF THEM
                variables = chromosome.variables.flatten()

                votes[rule.consequent] += self._rule_match(rule, x, chromosome.rules[i], chromosome.modifiers[i], variables)
        
            ## Then, we get the class with the highest sum
            predictions.append(max(votes, key=votes.get))

        return accuracy_score(self.y, predictions)
    