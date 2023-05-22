from abc import ABC, abstractmethod
from teacher.fuzzy import FuzzyContinuousSet
import numpy as np

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

    def match(self, value):
        return int(value >= self.range[0] and value <= self.range[1])
    

class CategoricalAntecedent(Antecedent):
    def __init__(self, variable, values):
        self.variable = variable
        self.values = values
    
    def __repr__(self) -> str:
        return f"{self.variable} is {self.values}"
    
    def __hash__(self) -> int:
        return hash((self.variable, (tuple(value) for value in self.values)))
    
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, CategoricalAntecedent):
            return False
        return hash(self) == hash(o)
    
    def match(self, value):
        for v in self.values:
            a, b = v
            if not ((value == a) is b):
                return 0
        return 1


class FuzzyAntecedent(Antecedent):
    def __init__(self, variable, fuzzy_set):
        self.variable = variable
        self.fuzzy_set = fuzzy_set

    def __repr__(self) -> str:
        return f"{self.variable} is {self.fuzzy_set.name}"
    
    def __hash__(self) -> int:
        return hash((self.variable, self.fuzzy_set))
    
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, FuzzyAntecedent):
            return False
        return hash(self) == hash(o)
    
    def match(self, value):
        return self.fuzzy_set.membership(np.array([value]))[0]

class HelloWorldRule():
    def __init__(self):
        pass

    def print_hello_world(self):
        print("Hello World!")

class Rule():
    def __init__(self, antecedent, consequent) -> None:
        self.antecedent = antecedent
        self.consequent = consequent
    
    def __repr__(self) -> str:
        return f"{self.antecedent} -> {self.consequent}"
    
    def __hash__(self) -> int:
        return hash((tuple(self.antecedent), self.consequent))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Rule):
            return False
        return hash(self) == hash(o)

    def match(self, x):
        return min([antecedent.match(x[antecedent.variable]) for antecedent in self.antecedent])
    
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
    def __init__(self, antecedent, consequent, weight):
        super().__init__(antecedent, consequent)
        self.weight = weight
    
    def __repr__(self) -> str:
        return f"{self.weight} : {self.antecedent} -> {self.consequent}"
    
    def match(self, x):
        return super().match(x) * self.weight
    
    @staticmethod
    def from_json(json_rule, dataset_info):
        antecedent, consequent, weight, fuzzy_sets = json_rule
        rule_antecedent = []

        for fs in fuzzy_sets:
            if fs[0] in dataset_info['discrete']:
                rule_antecedent.append(CategoricalAntecedent(dataset_info['antecedent_order'][fs[0]], [[fs[1], True]]))
            else:
                rule_antecedent.append(FuzzyAntecedent(dataset_info['antecedent_order'][fs[0]], FuzzyContinuousSet(fs[1],fs[2])))
        
        return FuzzyRule(rule_antecedent, consequent, weight)