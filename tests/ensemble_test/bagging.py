import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier as skBaggingClassifier
from sklearn.ensemble import BaggingRegressor as skBaggingRegressor

from ensemble.bagging import BaggingClassifier, BaggingRegressor

# Bagging Classifier Test
X, y = load_breast_cancer(return_X_y=True)
bag_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
bag_clf.fit(X, y)
y_pred = bag_clf.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Custom Bagging Classifier Accuracy: {accuracy}")

sk_bag_clf = skBaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
sk_bag_clf.fit(X, y)
sk_y_pred = sk_bag_clf.predict(X)
sk_accuracy = accuracy_score(y, sk_y_pred)
print(f"sklearn Bagging Classifier Accuracy: {sk_accuracy}")

assert np.allclose(accuracy, sk_accuracy), "The custom Bagging Classifier accuracy does not match sklearn's."

# Bagging Regressor Test
X, y = fetch_california_housing(return_X_y=True)
bag_reg = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=50, random_state=42)
bag_reg.fit(X, y)
y_pred = bag_reg.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Custom Bagging Regressor MSE: {mse}")

sk_bag_reg = skBaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=50, random_state=42)
sk_bag_reg.fit(X, y)
sk_y_pred = sk_bag_reg.predict(X)
sk_mse = mean_squared_error(y, sk_y_pred)
print(f"sklearn Bagging Regressor MSE: {sk_mse}")

assert np.allclose(mse, sk_mse, 0.01), "The custom Bagging Regressor MSE does not match sklearn's."
