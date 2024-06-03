import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import cross_val_predict as skcross_val_predict
from model_selection.__validation import cross_val_predict

from ensemble.random_forest import RandomForestClassifier, RandomForestRegressor

from tree.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier as skRandomClassifier
from sklearn.ensemble import RandomForestRegressor as skRandomForestRegressor


# regression
X, y = load_diabetes(return_X_y=True)
clf = DecisionTreeRegressor()
ans1 = cross_val_predict(clf, X, y)

# classification
X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier()
ans1 = cross_val_predict(clf, X, y)
print("cross val done")



X, y = load_diabetes(return_X_y=True)
clf = skRandomForestRegressor(random_state=0)
ans1 = cross_val_predict(clf, X, y)
ans2 = skcross_val_predict(clf, X, y)
assert np.allclose(ans1, ans2)

X, y = load_iris(return_X_y=True)
clf = skRandomClassifier(random_state=0)
ans1 = cross_val_predict(clf, X, y)
ans2 = skcross_val_predict(clf, X, y)
assert np.array_equal(ans1, ans2)
ans1 = cross_val_predict(clf, X, y, method="predict_proba")
ans2 = skcross_val_predict(clf, X, y, method="predict_proba")
assert np.allclose(ans1, ans2)