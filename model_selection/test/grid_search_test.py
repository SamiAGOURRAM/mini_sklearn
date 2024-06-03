import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV

from model_selection.GridSearch import GridSearchCVClassifier, GridSearchCVRegressor


def compare_grid_search():
    # Classification test
    X_iris, y_iris = load_iris(return_X_y=True)
    param_grid_classifier = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

    # Using custom GridSearchCV
    custom_grid_search_classifier = GridSearchCVClassifier(SVC(), param_grid_classifier)
    custom_grid_search_classifier.fit(X_iris, y_iris)
    custom_best_params_classifier = custom_grid_search_classifier.best_params_
    custom_best_score_classifier = max(custom_grid_search_classifier.cv_results_['mean_test_score'])

    # Using sklearn GridSearchCV
    sklearn_grid_search_classifier = SklearnGridSearchCV(SVC(), param_grid_classifier, cv=5)
    sklearn_grid_search_classifier.fit(X_iris, y_iris)
    sklearn_best_params_classifier = sklearn_grid_search_classifier.best_params_
    sklearn_best_score_classifier = sklearn_grid_search_classifier.best_score_

    # Regression test
    X_diabeties, y_diabeties = load_diabetes(return_X_y=True)
    param_grid_regressor = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

    # Using custom GridSearchCV
    custom_grid_search_regressor = GridSearchCVRegressor(SVR(), param_grid_regressor)
    custom_grid_search_regressor.fit(X_diabeties, y_diabeties)
    custom_best_params_regressor = custom_grid_search_regressor.best_params_
    custom_best_score_regressor = max(custom_grid_search_regressor.cv_results_['mean_test_score'])

    # Using sklearn GridSearchCV
    sklearn_grid_search_regressor = SklearnGridSearchCV(SVR(), param_grid_regressor, cv=5)
    sklearn_grid_search_regressor.fit(X_diabeties, y_diabeties)
    sklearn_best_params_regressor = sklearn_grid_search_regressor.best_params_
    sklearn_best_score_regressor = sklearn_grid_search_regressor.best_score_

    # Compare results
    print("Classification Comparison:")
    print("Custom GridSearchCV Best Params:", custom_best_params_classifier)
    print("Sklearn GridSearchCV Best Params:", sklearn_best_params_classifier)
    print("Custom GridSearchCV Best Score:", custom_best_score_classifier)
    print("Sklearn GridSearchCV Best Score:", sklearn_best_score_classifier)

    print("\nRegression Comparison:")
    print("Custom GridSearchCV Best Params:", custom_best_params_regressor)
    print("Sklearn GridSearchCV Best Params:", sklearn_best_params_regressor)
    print("Custom GridSearchCV Best Score:", custom_best_score_regressor)
    print("Sklearn GridSearchCV Best Score:", sklearn_best_score_regressor)

if __name__ == "__main__":
    compare_grid_search()



