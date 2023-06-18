from abc import ABC, abstractmethod
from teacher.fuzzy import FuzzyContinuousSet
import numpy as np
import copy
from sklearn.utils.validation import check_array, check_X_y

class Antecedent(ABC):
    pass

class NumericAntecedent(Antecedent):
    def __init__(self, variable, range):
        self.variable = variable
        self.range = range
    
    def __repr__(self) -> str:
        return f"{self.variable} is {self.range}"
    
    def __hash__(self) -> int:
        return hash((self.variable, tuple(self.range)))
    
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, NumericAntecedent):
            return False
        return hash(self) == hash(o)

    def __lt__(self, o: object) -> bool:
        return self.variable < o.variable

    def match(self, value):
        return int(value >= self.range[0] and value <= self.range[1])
    

class CategoricalAntecedent(Antecedent):
    def __init__(self, variable, values, operator = 'and'):
        self.variable = variable
        self.values = self._simplify_values(values)
        self.operator = operator
    
    def __repr__(self) -> str:
        return f"{self.variable} is {self.values}"
    
    def __hash__(self) -> int:
        return hash((self.variable, tuple(tuple(value) for value in self.values)))
    
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, CategoricalAntecedent):
            return False
        return hash(self) == hash(o)
    
    def __lt__(self, o: object) -> bool:
        return self.variable < o.variable
    
    def _simplify_values(self, values):
        unique_values = set([])
        vals = []
        for value in values:
            if tuple(value) not in unique_values:
                unique_values.add(tuple(value))
                vals.append(value)
        return vals

    def match(self, value):
        if self.operator == 'and':
            for v in self.values:
                a, b = v
                if not ((value == a) is b):
                    return 0
            return 1
        else:
            for v in self.values:
                a, b = v
                if (value == a) is b:
                    return 1
            return 0

class FuzzyAntecedent(Antecedent):
    def __init__(self, variable, fuzzy_set, modifier=None, multiple_sets=False):
        self.variable = variable
        self.fuzzy_set = fuzzy_set
        self.multiple_sets = multiple_sets
        self.modifier = modifier
        self._hash = None

    def __repr__(self) -> str:
        return f"{self.variable} is {self.modifier if self.modifier else ''} {self.fuzzy_set.name}"
    
    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self.variable, self.fuzzy_set, self.modifier))
        return self._hash
    
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, FuzzyAntecedent):
            return False
        return hash(self) == hash(o)

    def __lt__(self, o: object) -> bool:
        return self.variable < o.variable
    
    def match(self, value):
        match = 0
        if self.multiple_sets:
            match = np.sum(self.fuzzy_set.membership(np.array([value]))[0])
        else:
            match = self.fuzzy_set.membership(np.array([value]))[0]
        
        if self.modifier == 'very':
            return match ** 2
        elif self.modifier == 'slightly':
            return np.sqrt(match)
        else:
            return match

class HelloWorldRule():
    def __init__(self):
        pass

    def print_hello_world(self):
        print("Hello World!")

class Rule():
    def __init__(self, antecedent, consequent) -> None:
        self.antecedent = tuple(sorted(antecedent))
        self.consequent = consequent
        self.cache = {}
        self._hash = None
    
    def __repr__(self) -> str:
        return f"{self.antecedent} -> {self.consequent}"
    
    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self.antecedent, self.consequent))
        return self._hash

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Rule):
            return False
        return hash(self) == hash(o)

    def match(self, x):
        x = tuple(x)
        if x in self.cache:
            return self.cache[x]
        else:
            self.cache[x] = self._match(x)
            return self.cache[x]
    
    def _match(self, x):
        try:
            return min([antecedent.match(x[antecedent.variable]) for antecedent in self.antecedent])
        except:
            return 0

    def size(self):
        return len(self.antecedent)
    
    @staticmethod
    def from_json(json_rule, dataset_info):
        antecedent, consequent = json_rule
        rule_antecedent = []

        for ante in antecedent:
            if ante in dataset_info['discrete']:
                rule_antecedent.append(CategoricalAntecedent(dataset_info['antecedent_order'][ante], antecedent[ante]))
            else:
                rule_antecedent.append(NumericAntecedent(dataset_info['antecedent_order'][ante], antecedent[ante]))
        
        return Rule(rule_antecedent, consequent)
        


class FuzzyRule(Rule):
    def __init__(self, antecedent, consequent, weight=1):
        super().__init__(antecedent, consequent)
        self.weight = weight
    
    def __repr__(self) -> str:
        return f"{self.weight} : {self.antecedent} -> {self.consequent}"
    
    def match(self, x):
        return super().match(x) * self.weight
    
    def support(self, X):
        X = check_array(X, dtype=['float64', 'object'])
        return self._match_sum(X) / len(X)

    def confidence(self, X, y):
        X, y = check_X_y(X, y, dtype=['float64', 'object'])
        match_sum = self._match_sum(X)
        if not match_sum: 
            return 0
        return self._match_sum(np.array([x for x, y in zip(X, y) if y == self.consequent])) / match_sum

    def _match_sum(self, X):
        X = check_array(X, dtype=['float64', 'object'])
        return np.sum([super(FuzzyRule, self).match(x) for x in X])

    def update(self, update):
        new_rule = copy.deepcopy(self)
        for ante in new_rule.antecedent:
            if ante.variable in update:
                ante.fuzzy_set = update[ante.variable]
        return new_rule
    
    def fusion(self, rule2, weight='sum'):
        new_antecedent = []
        for a, b in zip(self.antecedent, rule2.antecedent):
            if isinstance(a, FuzzyAntecedent) and isinstance(b, FuzzyAntecedent):
                new_antecedent.append(FuzzyAntecedent(a.variable, FuzzyContinuousSet.merge(a.fuzzy_set, b.fuzzy_set)))
            else:
                new_antecedent.append(CategoricalAntecedent(a.variable, a.values + b.values, operator='or'))
        
        if weight == 'sum':
            new_weight = self.weight + rule2.weight
        elif weight == 'max':
            new_weight = max(self.weight, rule2.weight)
        else:
            raise ValueError(f"Weight {weight} not supported")

        return FuzzyRule(new_antecedent, self.consequent, new_weight)
    
    @staticmethod
    def from_json(json_rule, dataset_info, multiple_antecedents=False):
        antecedent, consequent, weight, fuzzy_sets = json_rule
        rule_antecedent = []

        if multiple_antecedents:
            for ante in antecedent:
                if ante in dataset_info['discrete']:
                    rule_antecedent.append(CategoricalAntecedent(dataset_info['antecedent_order'][ante], [[value, True] for value in antecedent[ante]], operator='or'))
                else:
                    rule_antecedent.append(FuzzyAntecedent(dataset_info['antecedent_order'][ante], [FuzzyContinuousSet(fs[1],fs[2]) for fs in antecedent[ante]], multiple_sets = True))
        else:
            for fs in fuzzy_sets:
                if fs[0] in dataset_info['discrete']:
                    rule_antecedent.append(CategoricalAntecedent(dataset_info['antecedent_order'][fs[0]], [[fs[1], True]]))
                else:
                    rule_antecedent.append(FuzzyAntecedent(dataset_info['antecedent_order'][fs[0]], FuzzyContinuousSet(fs[1],fs[2])))
            
        return FuzzyRule(rule_antecedent, consequent, weight)
    
    def chromosome(self, metadata):
        chromosome = np.ones(len(metadata['fuzzy_variables_order'])+1)
        chromosome *= -1
        for premise in self.antecedent:
            if isinstance(premise, CategoricalAntecedent):
                if len(premise.values) > 1:
                    raise ValueError("Categorical Antecedent with multiple values not supported")
                chromosome[premise.variable] = metadata[premise.variable]['values'][premise.values[0][0]]
            elif isinstance(premise, FuzzyAntecedent):
                chromosome[premise.variable] = metadata[premise.variable]['points'][float(premise.fuzzy_set.name)]
        chromosome[-1] = self.consequent
        return chromosome
    
    def modifier_chromosome(self, metadata):
        chromosome = np.ones(len(metadata['continuous']))
        chromosome *= -1
        for premise in self.antecedent:
            if premise.variable in metadata['continuous']:
                if premise.modifier == 'very':
                    chromosome[metadata['continuous'][premise.variable]] = 0
                elif premise.modifier == 'slightly':
                    chromosome[metadata['continuous'][premise.variable]] = 1            
        return chromosome
    
    @staticmethod
    def from_chromosome(rule_chromosome, modifier_chromosome, fuzzy_variables, metadata, all_antecedents):
        antecedents = []
        for i, var in enumerate(fuzzy_variables):
            if rule_chromosome[i] == -1: # If fuzzy variable not in use
                continue
            if (i, rule_chromosome[i]) in all_antecedents: # Check if we already have this antecedent
                antecedents.append(all_antecedents[(i, rule_chromosome[i])])
                continue
            if isinstance(var.fuzzy_sets[0], FuzzyContinuousSet): # If not we create it
                new_ante = FuzzyAntecedent(i, var.fuzzy_sets[int(rule_chromosome[i])])
            else:
                new_ante = CategoricalAntecedent(i, [[var.fuzzy_sets[int(rule_chromosome[i])].value, True]])
            all_antecedents[(i, rule_chromosome[i])] = new_ante # And save it
            antecedents.append(new_ante)
        
        for premise in antecedents: # Add modifiers
            if premise.variable in metadata['continuous']:
                if modifier_chromosome[metadata['continuous'][premise.variable]] == 0:
                    premise.modifier = 'very'

                elif modifier_chromosome[metadata['continuous'][premise.variable]] == 1:
                    premise.modifier = 'slightly'
        
        return FuzzyRule(antecedents, rule_chromosome[-1])