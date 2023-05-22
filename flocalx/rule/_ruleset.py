from ..rule import Rule, FuzzyRule
from sklearn.utils import check_array, check_X_y
import numpy as np


class RuleSet():
    def __init__(self, rules):
        self.rules = rules

    @staticmethod
    def from_json(json_ruleset, dataset_info):
        rules = set([])

        for rule in json_ruleset:
            rules.add(Rule.from_json(rule, dataset_info))
    
        return RuleSet(rules)

    def fit(self, X, y):
        pass

    def predict(self, X):
        X = check_array(X, dtype=['float64', 'object'])
        return [max([(rule.match(x), rule.consequent) for rule in self.rules])[1] for x in X]
    
    def score(self, X, y):
        X, y = check_X_y(X, y, dtype=['float64', 'object'])
        return np.sum(self.predict(X) == y)/y.shape[0]

    def size(self):
        return len(self.rules)
    
    def rule_size(self):
        return np.mean([rule.size() for rule in self.rules])

class FuzzyRuleSet(RuleSet):
    def __init__(self, rules):
        self.rules = rules

    @staticmethod
    def from_json(json_ruleset, dataset_info):
        rules = set([])

        for rule in json_ruleset:
            rules.add(FuzzyRule.from_json(rule, dataset_info))
    
        return FuzzyRuleSet(rules)
    
    def predict(self, X):
        X = check_array(X, dtype=['float64', 'object'])

        ## Aggregated vote
        ## First, we get the sum of the match values for each class
        predictions = []
        votes = {}
        for x in X:
            for rule in self.rules:
                if rule.consequent not in votes:
                    votes[rule.consequent] = 0
                votes[rule.consequent] += rule.match(x)
        
            ## Then, we get the class with the highest sum
            predictions.append(max(votes, key=votes.get))
        print(votes)
        return predictions
    
    def _robust_threshold(self, instance, rule_list, class_val):
        """Obtain the robust threshold"""
        other_classes = np.unique([rule.consequent for rule in rule_list if rule.consequent != class_val])
        all_th = []
        for cv in other_classes:
            th = 0
            for rule in rule_list:
                if rule.consequent == cv:
                    th += rule.match(instance) * rule.weight

            all_th.append(th)

        return max(all_th)
    
    def mr_factual(self, x, threshold = 0.001):
        """
        Generate the minimum robust factual.

        Parameters
        ----------
        instance : dict, of format {set_1: pert_1, set_2: pert_2, ...}, ...}
            Fuzzy representation of the instance with all the features and pertenence
            degrees to each fuzzy set
        rule_list : list[Rule]
            List of candidate rules to form part of the factual
        class_val : str
            Predicted value that the factual will explain

        Returns
        -------
        list[Rule]
            List of factual rules
        """
        class_val = self.predict(x.reshape(1, -1))[0]
        print(class_val)
        fired_rules =  [rule for rule in self.rules if rule.match(x) > threshold]
        print(fired_rules)
        class_fired_rules = [rule for rule in fired_rules if rule.consequent == class_val]
        class_fired_rules.sort(key=lambda rule: rule.match(x) * rule.weight, reverse=True)
        robust_threshold = self._robust_threshold(x, self.rules, class_val)

        factual = []
        AD_sum = 0
        for rule in class_fired_rules:
            if robust_threshold < AD_sum:
                break
            factual.append(rule)
            AD_sum += rule.match(x) * rule.weight
        return factual


class FLocalX(FuzzyRuleSet):
    def __init__(self, rules, merge_operators):
        self.rules = rules
        self.merge_operators = merge_operators

        self.MERGE_OPERATORS = {
            "variable_selection": self._variable_selection,
            "fuzzy_set_fusion": self._fuzzy_set_fusion,
            "variable_mapping": self._variable_mapping
        }


    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=['float64', 'object'])
        if self.rules == set([]):
            raise Exception("No rules to fit")

        for operator in self.merge_operators:
            self.MERGE_OPERATORS[operator](X, y)
    
    def _variable_selection(self, X, y):
        pass

    def _fuzzy_set_fusion(self, X, y):
        pass

    def _variable_mapping(self, X, y):
        pass