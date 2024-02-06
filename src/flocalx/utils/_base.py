from ._fuzzy_set import FuzzyContinuousSet, FuzzyDiscreteSet
from ._fuzzy_variable import FuzzyVariable


def get_fuzzy_variables(continuous_fuzzy_points, discrete_fuzzy_values, order,
                        continuous_labels=None, discrete_labels=None, point_set_method='point_set'):
    """Build the fuzzy variables given the points of the triangles that
    define them, as well as the values of the discrete variables.

    Parameters
    ----------
    continuous_fuzzy_points : dict
        Dictionary with format {feature: [peak_1, peak_2, ...]} with the name
        of the features and the peaks of the triangles of each fuzzy set.
    discrete_fuzzy_values : dict
        Dictionary with format {feature : [value_1, value_2, ...]} with the name
        of the features and the unique values thatthe discrete variable can take
    order : dict
        Dictionary with format {name : position} where each name is the label
        of the fuzzy variable and the position relative to an input dataset
    continuous_labels : dict, optional
        Dictionary with format {feature : [label_1, label_2, ...]} with the name
        the continuous variable and the labels of the fuzzy
        sets associated to the peaks peak_1, peak_2, ...
    discrete_labels : dict, optional
        Dictionary with format {feature : [label_1, label_2, ...]} with the name
        the discrete variable and the labels of the fuzzy
        sets associated to the values value_1, value_2, ...
    point_set : str, 'point_set' by default
        Method to generate the point sets. Defaults to `point_set`

    Returns
    -------
    list[FuzzyVariable]
        Ordered list of all the fuzzy variables
    """
    fuzzy_variables = [None] * len(order)
    for name, points in continuous_fuzzy_points.items():
        if continuous_labels is None or name not in continuous_labels:
            col_labels = [f'{label}' for label in continuous_fuzzy_points[name]]
        else:
            col_labels = continuous_labels[name]
        fuzzy_variables[order[name]] = FuzzyVariable(name, get_fuzzy_continuous_sets(list(zip(col_labels, points)),
                                                                                     point_set_method=point_set_method))

    for name, values in discrete_fuzzy_values.items():
        if discrete_labels is None or name not in discrete_labels:
            col_labels = [f'{label}' for label in discrete_fuzzy_values[name]]
        else:
            col_labels = discrete_labels[name]
        fuzzy_variables[order[name]] = FuzzyVariable(name, get_fuzzy_discrete_sets(list(zip(col_labels, values))))

    return fuzzy_variables


def _point_set(divisions):
    """Generate a FuzzyContinuousSet of a single point
    with all three values the same

    Parameters
    ----------
    divisions : tuple
        Tuple with the name of the set and the peak of the triangle
        like ('low', 0)
    """
    return [FuzzyContinuousSet(divisions[0], [divisions[1], divisions[1], divisions[1]], point_set=True)]


def get_fuzzy_continuous_sets(divisions, point_set_method='point_set'):
    """Generate a list with the triangular fuzzy sets of
    a variable of a DataFrame given the peaks of
    the triangles

    Parameters
    ----------
    divisions : list
        List of tuples with the names of the sets and the peak of the triangle
        like [('low', 0), ('mid', 2), ('high', 5)]
    point_set_method : str, 'point_set' by default
        Name of the method to generate the point sets.
        Defaults to `point_set`

    Returns
    -------
    list
        List with all the Fuzzy Continuous Sets that form a Fuzzy Variable
    """
    # WE FIRST CHECK IF IT IS A POINT VALUE
    if len(divisions) == 1:
        if point_set_method == 'point_set':
            return _point_set(divisions[0])
        else:
            raise ValueError(f'Point set method {point_set_method} is not valid')

    fuzzy_sets = []
    fuzzy_sets.append(FuzzyContinuousSet(divisions[0][0], [divisions[0][1], divisions[0][1], divisions[1][1]]))
    # First triangle is only half triangle

    for i in range(len(divisions) - 2):
        fuzzy_sets.append(FuzzyContinuousSet(divisions[i + 1][0], [divisions[i][1],
                                                                   divisions[i + 1][1], divisions[i + 2][1]]))

    # Last triangle is only half triangle
    fuzzy_sets.append(FuzzyContinuousSet(divisions[-1][0], [divisions[-2][1], divisions[-1][1], divisions[-1][1]]))

    return fuzzy_sets


def get_fuzzy_discrete_sets(divisions):
    """Generate a list with the discrete fuzzy sets of
    a variable of a DataFrame given the unique values
    it can take

    Parameters
    ----------
    divisions : list
        List of tuples with the names of the sets and the peak of the triangle
        like [('low', 0), ('mid', 2), ('high', 5)]

    Returns
    -------
    list
        List with all the Fuzzy Discrete Sets that form a Fuzzy Variable
    """

    return [FuzzyDiscreteSet(name, value) for name, value in divisions]
