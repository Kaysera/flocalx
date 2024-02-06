[![codecov](https://codecov.io/gh/Kaysera/flocalx/branch/main/graph/badge.svg?token=QFA17A64EW)](https://codecov.io/gh/Kaysera/flocalx)
[![GitHub - License](https://img.shields.io/github/license/Kaysera/flocalx?logo=github&style=flat&color=green)](https://github.com/Kaysera/flocalx/blob/main/LICENSE)
[![Lint](https://github.com/Kaysera/flocalx/actions/workflows/linting.yml/badge.svg)](https://github.com/Kaysera/flocalx/actions/workflows/linting.yml)


# FLocalX: Fuzzy Global through Local Explainability

Explanations come in two forms: local, explaining a single model prediction, and global, explaining all model predictions. 
The Local to Global (L2G) problem consists of bridging these two families of explanations. 
Simply put, we generate global explanations by merging local ones.

FLocalX is an open source Python Library that provides a framework to explore the creation
of global explanations derived from local explanations in the form of rulesets. The 
objective of the library is to be extensible with new explainers and metaheuristics 
approaches to create new global explanations.

## Installation

### Dependencies

FLocalX requires:

    * Python (>=3.9)
    * NumPy 
    * Scikit-Learn

### User installation

If you already have a working installation, you can install FLocalX with 

```shell
git clone https://github.com/Kaysera/flocalx
pip install flocalx
```

## Usage

For detailed instructions on how to use FLocalX, please refer to the examples folder

## Supported Methods

The following explainers are currently supported:
- **LORE**: Local explainer generated from a neighborhood
- **FLARE**: Fuzzy local explainer generated from a neighborhood

The following metaheuristics are currently supported:
- **Genetic Algorithm**