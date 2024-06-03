# Custom Machine Learning Library

## Overview

This project is a custom implementation of various machine learning algorithms and utilities. The aim is to provide a deeper understanding of how these algorithms work by implementing them from scratch. The library includes modules for preprocessing, feature selection, ensemble methods, and neural networks.

## Modules

### base
- **BaseEstimator**: Base class for all estimators.
- **TransformerMixin**: Mixin class for all transformers.
- **ClassifierMixin**: Mixin class for all classifiers.
- **RegressorMixin**: Mixin class for all regressors.

### preprocessing
- **StandardScaler**: Standardizes features by removing the mean and scaling to unit variance.
- **MinMaxScaler**: Transforms features by scaling each feature to a given range.
- **Normalizer**: Normalizes samples individually to unit norm.

### encoding
- **LabelEncoder**: Encodes target labels with value between 0 and n_classes-1.
- **OneHotEncoder**: Encodes categorical features as a one-hot numeric array.
- **OrdinalEncoder**: Encodes categorical features as an integer array.

### feature_selection
- **VarianceThreshold**: Feature selector that removes all low-variance features.
- **SelectFromModel**: Meta-transformer for selecting features based on importance weights.
- **RFE (Recursive Feature Elimination)**: Selects features by recursively considering smaller and smaller sets of features.

### ensemble
- **Bagging**: Implements bagging ensemble method.
- **RandomForestRegressor**: Random forest regressor implementation.
- **VotingClassifier**: Soft and hard voting for classification.
- **VotingRegressor**: Voting for regression.
- **StackingClassifier**: Stacking ensemble for classification.
- **StackingRegressor**: Stacking ensemble for regression.

### neural_network
- **NeuralNetwork**: A basic implementation of a feedforward neural network.
- **MLPClassifier**: Multi-layer perceptron classifier.
- **MLPRegressor**: Multi-layer perceptron regressor.

### model_selection
- **train_test_split**: Split arrays or matrices into random train and test subsets.
- **cross_validation**
- **KFold**
- **StratifiedKFold**

### utils
- **_encode**: Utility function for encoding.
- **_unique**: Utility function for finding unique values.
- **column_or_1d**: Utility function to ensure data is 1-dimensional.
- **_num_samples**: Utility function to get the number of samples in an array-like object.

## Installation

To use this library, simply clone this repository and import the desired modules in your Python scripts.

```bash
git clone https://github.com/yourusername/custom_ml_library.git
