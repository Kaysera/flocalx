import numpy as np
from teacher.fuzzy import get_fuzzy_variables
import flocalx.rule

class Chromosome:
    def __init__(self, variables, rules, modifiers, used_rules, alpha = 0.8, fitness = lambda x: 0):
        self.variables = variables
        self.rules = rules
        self.modifiers = modifiers
        self.used_rules = used_rules
        self.genes = np.concatenate([variables.flatten(), rules.flatten(), modifiers.flatten(), used_rules.flatten()])
        self.alpha = alpha

        # Fitness es una funcion que toma como parametro el cromosoma y devuelve un valor numerico
        self.fitness = fitness
        self.score = fitness(self)

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
    
    def _fuzzy_variables(self, metadata):
        fuzzy_points = {}
        for order in metadata['continuous']:
            fuzzy_points[metadata[order]['name']] = [metadata[order]['min']] + sorted(self.variables[metadata['continuous'][order]]) + [metadata[order]['max']]
        return get_fuzzy_variables(fuzzy_points, metadata['discrete_fuzzy_values'], metadata['fuzzy_variables_order'])

    def to_rule_based_system(self, metadata):
        fuzzy_variables = self._fuzzy_variables(metadata)
        rules = set([])

        for rule, modifiers, used in zip(self.rules, self.modifiers, self.used_rules):
            if used:
                rule = flocalx.rule.FuzzyRule.from_chromosome(rule, modifiers, fuzzy_variables, metadata)
                rules.add(rule)
        
        return flocalx.rule.FLocalX(rules)
    
    def _max_min_arithmetic_crossover(self, parent1, parent2):
        child1 = np.min([parent1, parent2], axis = 0)
        child2 = np.max([parent1, parent2], axis = 0)
        child3 = self.alpha * parent1 + (1 - self.alpha) * parent2
        child4 = self.alpha * parent2 + (1 - self.alpha) * parent1
        return child1, child2, child3, child4
    
    def _two_point_crossover(self, parent1, parent2):
        a, b = sorted(np.random.randint(0, len(parent1), 2))

        child1 = np.concatenate([parent1[:a], parent2[a:b], parent1[b:]])
        child2 = np.concatenate([parent2[:a], parent1[a:b], parent2[b:]])

        return child1, child2
    
    def crossover(self, other):

        variables_child_1, variables_child_2, variables_child_3, variables_child_4 = self._max_min_arithmetic_crossover(self.variables.flatten(), other.variables.flatten())
        rules_child_1, rules_child_2 = self._two_point_crossover(self.rules.flatten(), other.rules.flatten())
        modifiers_child_1, modifiers_child_2 = self._two_point_crossover(self.modifiers.flatten(), other.modifiers.flatten())
        used_rules_child_1, used_rules_child_2 = self._two_point_crossover(self.used_rules.flatten(), other.used_rules.flatten())

        child_1 = Chromosome(variables_child_1.reshape(self.variables.shape), rules_child_1.reshape(self.rules.shape), modifiers_child_1.reshape(self.modifiers.shape), used_rules_child_1.reshape(self.used_rules.shape), self.alpha, self.fitness)
        child_2 = Chromosome(variables_child_1.reshape(self.variables.shape), rules_child_2.reshape(self.rules.shape), modifiers_child_2.reshape(self.modifiers.shape), used_rules_child_2.reshape(self.used_rules.shape), self.alpha, self.fitness)

        child_3 = Chromosome(variables_child_2.reshape(self.variables.shape), rules_child_1.reshape(self.rules.shape), modifiers_child_1.reshape(self.modifiers.shape), used_rules_child_1.reshape(self.used_rules.shape), self.alpha, self.fitness)
        child_4 = Chromosome(variables_child_2.reshape(self.variables.shape), rules_child_2.reshape(self.rules.shape), modifiers_child_2.reshape(self.modifiers.shape), used_rules_child_2.reshape(self.used_rules.shape), self.alpha, self.fitness)

        child_5 = Chromosome(variables_child_3.reshape(self.variables.shape), rules_child_1.reshape(self.rules.shape), modifiers_child_1.reshape(self.modifiers.shape), used_rules_child_1.reshape(self.used_rules.shape), self.alpha, self.fitness)
        child_6 = Chromosome(variables_child_3.reshape(self.variables.shape), rules_child_2.reshape(self.rules.shape), modifiers_child_2.reshape(self.modifiers.shape), used_rules_child_2.reshape(self.used_rules.shape), self.alpha, self.fitness)

        child_7 = Chromosome(variables_child_4.reshape(self.variables.shape), rules_child_1.reshape(self.rules.shape), modifiers_child_1.reshape(self.modifiers.shape), used_rules_child_1.reshape(self.used_rules.shape), self.alpha, self.fitness)
        child_8 = Chromosome(variables_child_4.reshape(self.variables.shape), rules_child_2.reshape(self.rules.shape), modifiers_child_2.reshape(self.modifiers.shape), used_rules_child_2.reshape(self.used_rules.shape), self.alpha, self.fitness)

        return sorted([child_1, child_2, child_3, child_4, child_5, child_6, child_7, child_8], reverse = True)[:2]

    def _variables_mutation(self, metadata):
        new_variables = np.copy(self.variables)
        m, n = np.random.randint(self.variables.shape[0]), np.random.randint(self.variables.shape[1])
        v = list(metadata['continuous'].keys())[m]
        min_val, max_val = metadata[v]['min'], metadata[v]['max']
        new_val = np.random.uniform(min_val, max_val)
        new_variables[m, n] = new_val
        return new_variables
    
    def _rules_mutation(self, metadata):
        new_rules = np.copy(self.rules)
        m, n = np.random.randint(self.rules.shape[0]), np.random.randint(self.rules.shape[1]-2)
        if 'values' in metadata[n]:
            length = len(metadata[n]['values'])
        else:
            length = len(metadata[n]['points'])
        new_rules[m, n] = np.random.randint(-1, length)
        return new_rules
    
    def _modifiers_mutation(self):
        new_modifiers = np.copy(self.modifiers)
        m, n = np.random.randint(self.modifiers.shape[0]), np.random.randint(self.modifiers.shape[1])
        new_modifiers[m, n] = np.random.randint(-1, 2)
        return new_modifiers
    
    def _used_rules_mutation(self):
        new_used_rules = np.copy(self.used_rules)
        m = np.random.randint(self.used_rules.shape[0])
        new_used_rules[m] = 1 - new_used_rules[m]
        return new_used_rules
    
    def mutation(self, metadata):
        new_variables = self._variables_mutation(metadata)
        new_rules = self._rules_mutation(metadata)
        new_modifiers = self._modifiers_mutation()
        new_used_rules = self._used_rules_mutation()
        return Chromosome(new_variables, new_rules, new_modifiers, new_used_rules, self.alpha, self.fitness)
    
    def generate_initial_population(self, metadata, population_size):
        population = []
        pop_size = population_size // 4
        for i in range(pop_size):
            new_variables = self.random_variables(metadata)
            population.append(Chromosome(new_variables, self.rules, self.modifiers, self.used_rules, self.alpha, self.fitness))
        
        for i in range(pop_size):
            new_rules = self.random_rules(metadata)
            population.append(Chromosome(self.variables, new_rules, self.modifiers, self.used_rules, self.alpha, self.fitness))

        for i in range(pop_size):
            new_modifiers = self.random_modifiers()
            population.append(Chromosome(self.variables, self.rules, new_modifiers, self.used_rules, self.alpha, self.fitness))

        for i in range(pop_size):
            new_used_rules = self.random_used_rules()
            population.append(Chromosome(self.variables, self.rules, self.modifiers, new_used_rules, self.alpha, self.fitness))

        return population
    
    def random_variables(self, metadata):
        new_variables = np.copy(self.variables)
        for i in range(self.variables.shape[0]):
            for j in range(self.variables.shape[1]):
                v = list(metadata['continuous'].keys())[i]
                min_val, max_val = metadata[v]['min'], metadata[v]['max']
                new_variables[i, j] = np.random.uniform(min_val, max_val)
        return new_variables
    
    def random_rules(self, metadata):
        new_rules = np.copy(self.rules)
        for i in range(self.rules.shape[0]):
            for j in range(self.rules.shape[1]-2):
                if 'values' in metadata[j]:
                    length = len(metadata[j]['values'])
                else:
                    length = len(metadata[j]['points'])
                new_rules[i, j] = np.random.randint(-1, length)
        return new_rules
    
    def random_modifiers(self):
        new_modifiers = np.copy(self.modifiers)
        for i in range(self.modifiers.shape[0]):
            for j in range(self.modifiers.shape[1]):
                new_modifiers[i, j] = np.random.randint(-1, 2)
        return new_modifiers
    
    def random_used_rules(self):
        new_used_rules = np.copy(self.used_rules)
        for i in range(self.used_rules.shape[0]):
            new_used_rules[i] = np.random.randint(0, 2)
        return new_used_rules