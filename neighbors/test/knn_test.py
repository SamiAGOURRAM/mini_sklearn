
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as skKNeighborsClassifier

from neighbors.KNeighborsClassifier import KNeighborsClassifier

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
clf1 = KNeighborsClassifier().fit(X, y)
clf2 = skKNeighborsClassifier().fit(X, y)
prob1 = clf1.predict_proba(X)
prob2 = clf2.predict_proba(X)
assert np.allclose(prob1, prob2)
pred1 = clf1.predict(X)
pred2 = clf2.predict(X)
assert np.array_equal(pred1, pred2)