# =============================================================================
# Imports
# =============================================================================

# Standard library
import copy

# Third party
import numpy as np
from sklearn.utils import check_array, check_X_y
from sklearn.metrics import roc_auc_score
from ..utils import FuzzyContinuousSet

# Local application
from ..rule import Rule, FuzzyRule, FuzzyAntecedent

# =============================================================================
# Classes
# =============================================================================


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

    def auc(self, X, y):
        X, y = check_X_y(X, y, dtype=['float64', 'object'])
        return roc_auc_score(y, self.predict(X))

    def size(self):
        return len(self.rules)

    def rule_size(self):
        return np.mean([rule.size() for rule in self.rules])


class FuzzyRuleSet(RuleSet):
    def __init__(self, rules, max_class=None, random_state=None):
        self.rules = rules
        self.max_class = max_class
        if random_state is None:
            self.random_state = np.random.default_rng()
        else:
            self.random_state = random_state

    @staticmethod
    def from_json(json_ruleset, dataset_info):
        rules = set([])

        for rule in json_ruleset:
            rules.add(FuzzyRule.from_json(rule, dataset_info))

        return FuzzyRuleSet(rules)

    def predict_proba(self, X):
        X = check_array(X, dtype=['float64', 'object'])

        # Aggregated vote
        # First, we get the sum of the match values for each class
        predictions = np.zeros([len(X), 2])
        for i, x in enumerate(X):
            votes = {}
            for rule in self.rules:
                if rule.consequent not in votes:
                    votes[rule.consequent] = 0
                votes[rule.consequent] += rule.match(x)

            # Then, we get the class with the highest sum
            if votes and max(votes.values()) > 0:
                total_sum = sum(votes.values())
                pred = np.zeros(len(votes))
                for k, v in votes.items():
                    pred[k] = v/total_sum
                predictions[i] = pred
            else:
                if self.max_class is None:
                    predictions[i] = np.array([0.5, 0.5])
                else:
                    predictions[i] = np.array([0 if i != self.max_class else 1 for i in range(self.max_class+1)])
        return np.array(predictions)

    def predict(self, X):
        return np.argmax(self.predict_proba(self, X), axis=1)

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

    def mr_factual(self, x, threshold=0.001):
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
        fired_rules = [rule for rule in self.rules if rule.match(x) > threshold]
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
    def __init__(self, rules, max_class=None, merge_operators=[], random_state=None):
        if random_state is None:
            self.random_state = np.random.default_rng()
        else:
            self.random_state = random_state
        self.rules = rules
        self.max_class = max_class
        self.merge_operators = merge_operators

        self.MERGE_OPERATORS = {
            "variable_selection": self._variable_selection,
            "fuzzy_set_fusion": self._fuzzy_set_fusion,
            "similar_rule_fusion": self._similar_rule_fusion,
        }

    @staticmethod
    def from_json(json_ruleset, dataset_info, merge_operators=[], random_state=None):
        if random_state is None:
            random_state = np.random.default_rng()
        else:
            random_state = random_state
        rules = set([])
        if 'max_class' in dataset_info:
            max_class = dataset_info['max_class']
        else:
            max_class = None

        for rule in json_ruleset:
            rules.add(FuzzyRule.from_json(rule, dataset_info))

        return FLocalX(rules, max_class=max_class, merge_operators=merge_operators, random_state=random_state)

    @staticmethod
    def from_chromosome(chromosome, metadata, merge_operators=[], random_state=None):
        fuzzy_variables = chromosome._fuzzy_variables(metadata)
        rules = set([])
        all_antecedents = {}
        if 'max_class' in metadata:
            max_class = metadata['max_class']
        else:
            max_class = len(metadata['classes'])

        for rule, modifiers, used in zip(chromosome.rules, chromosome.modifiers, chromosome.used_rules):
            if used:
                rule = FuzzyRule.from_chromosome(rule,
                                                 modifiers,
                                                 fuzzy_variables,
                                                 metadata,
                                                 all_antecedents)
                rules.add(rule)

        return FLocalX(rules, max_class=max_class, merge_operators=merge_operators, random_state=random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=['float64', 'object'])
        if self.rules == set([]):
            raise Exception("No rules to fit")

        for operator in self.merge_operators:
            self.MERGE_OPERATORS[operator](X, y)

    def _variable_selection(self, X, y):
        pass

    def _fuzzy_set_fusion(self, X, y):
        # Extract all the fuzzy sets and the rules they are associated
        system_fuzzy_sets = self._extract_fuzzy_sets()
        # Fuse the similar fuzzy sets
        fused_fuzzy_sets = self._system_fuzzy_set_fusion(system_fuzzy_sets)
        # Update the rules with the new fuzzy sets
        rule_updates = self._extract_rule_dict(fused_fuzzy_sets)
        # Update the ruleset
        self.rules = self._update_ruleset(rule_updates)

    def _update_ruleset(self, rule_dict):
        new_ruleset = set([])
        for i, rule in enumerate(self.rules):
            if i in rule_dict:
                new_ruleset.add(rule.update(rule_dict[i]))
            else:
                new_ruleset.add(rule)
        return new_ruleset

    def _extract_rule_dict(self, system_dict):
        rules_dict = {}
        for variable in system_dict:
            for fuzzy_set in system_dict[variable]:
                for rule in system_dict[variable][fuzzy_set]:
                    if rule not in rules_dict:
                        rules_dict[rule] = {variable: fuzzy_set}
                    else:
                        rules_dict[rule][variable] = fuzzy_set
        return rules_dict

    def _system_fuzzy_set_fusion(self, system_dict, threshold=0.5):
        # Fuse the fuzzy sets with a similarity higher than threshold
        # Iterate until no sets are fused
        new_system_dict = {}

        for variable in system_dict:
            sets_dict = system_dict[variable]
            first_len = len(sets_dict.keys())
            new_sets_dict = self._similarity_fusion(sets_dict, threshold=threshold)
            second_len = len(new_sets_dict.keys())
            while first_len != second_len:
                first_len = second_len
                new_sets_dict = self._similarity_fusion(new_sets_dict)
                second_len = len(new_sets_dict.keys())

            new_system_dict[variable] = new_sets_dict

        return new_system_dict

    def _similarity_fusion(self, sets_dict, threshold=0.5):
        # Fuse the sets with a similarity higher than threshold
        new_sets_dict = {}
        sets = sorted(list(sets_dict.keys()))
        while len(sets) >= 2:
            a = sets.pop(0)
            b = sets.pop(0)
            sim = FuzzyContinuousSet.jaccard_similarity(a, b)
            if sim > threshold:
                new_sets_dict[FuzzyContinuousSet.merge(a, b)] = sets_dict[a] + sets_dict[b]

            else:
                new_sets_dict[a] = sets_dict[a]
                new_sets_dict[b] = sets_dict[b]

        for s in sets:
            new_sets_dict[s] = sets_dict[s]

        return new_sets_dict

    def _extract_fuzzy_sets(self):
        fss = {}
        for i, rule in enumerate(self.rules):
            for antecedent in rule.antecedent:
                if isinstance(antecedent, FuzzyAntecedent):
                    if antecedent.variable not in fss:
                        fss[antecedent.variable] = {antecedent.fuzzy_set: [i]}
                    elif antecedent.fuzzy_set not in fss[antecedent.variable]:
                        fss[antecedent.variable][antecedent.fuzzy_set] = [i]
                    else:
                        fss[antecedent.variable][antecedent.fuzzy_set].append(i)
        return fss

    def _similar_rule_fusion(self, X, y):
        # Group the rules with the same variables in the antecedent
        same_antecedent_rules = self._group_rules_by_antecedent()

        # Combine the rules with the same antecedent
        combined_rules = [
            self._combine_rules_with_same_antecedent([list(self.rules)[i] for i in same_antecedent_rules[k]], X, y)
            for k in same_antecedent_rules
        ]

        # Remove the rules before combining
        obsolete_rules = []
        for k in same_antecedent_rules:
            obsolete_rules += same_antecedent_rules[k]

        new_ruleset = [r for i, r in enumerate(self.rules) if i not in obsolete_rules]

        # Add the new combined rules
        for group in combined_rules:
            for rule in group:
                new_ruleset.append(rule)

        self.rules = set(new_ruleset)

    def _group_rules_by_antecedent(self):
        grouped_rules = {}
        for i, rule in enumerate(self.rules):
            key = (tuple(sorted([x.variable for x in rule.antecedent])), rule.consequent)
            if key not in grouped_rules:
                grouped_rules[key] = [i]
            else:
                grouped_rules[key].append(i)
        return {k: v for k, v in grouped_rules.items() if len(v) > 1}

    def _improves(self, first, second, fusion, X, y, loss=0.95):
        return fusion.confidence(X, y) > loss * max(first.confidence(X, y), second.confidence(X, y))

    def _combine_rules_with_same_antecedent(self, ruleset, X, y):
        changes = True
        i = 0
        while changes and len(ruleset) > 1:
            # print('Iteration', i)
            ruleset = sorted(ruleset, key=lambda x: (x.support(X), x.confidence(X, y)), reverse=True)
            # print(ruleset)
            first = ruleset.pop(0)
            new_ruleset = []
            changes = False
            while ruleset:
                try:
                    second = ruleset.pop(0)
                except Exception:  # No more rules in ruleset to merge
                    # print('Breaking')
                    new_ruleset.append(first)
                    break
                # print(f"Fusing rules {first} and {second}")
                fusion = first.fusion(second)
                if self._improves(first, second, fusion, X, y):
                    # print(f'Fusion improves: {fusion}')
                    changes = True
                    new_ruleset.append(fusion)
                    if len(ruleset) > 1:
                        first = ruleset.pop(0)
                    else:
                        new_ruleset += ruleset
                        break
                    # print(f'New first: {first}')
                else:
                    if len(ruleset) > 0:
                        new_ruleset.append(first)
                        first = second
                    else:
                        new_ruleset.append(first)
                        new_ruleset.append(second)
                        break

            ruleset = new_ruleset
            i += 1

        return ruleset

    def variable_mapping(self, new_fuzzy_variables):
        new_ruleset = set([])

        for rule in self.rules:
            new_rule = copy.deepcopy(rule)
            for premise in new_rule.antecedent:
                if isinstance(premise, FuzzyAntecedent):
                    premise.fuzzy_set = max(new_fuzzy_variables[premise.variable].fuzzy_sets,
                                            key=lambda set: FuzzyContinuousSet.jaccard_similarity(set,
                                                                                                  premise.fuzzy_set))
            new_ruleset.add(new_rule)
        self.rules = new_ruleset
