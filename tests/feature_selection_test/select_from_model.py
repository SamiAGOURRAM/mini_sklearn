import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import load_breast_cancer
from linear_model.logistic_regression import LogisticRegression
from sklearn.feature_selection import SelectFromModel as skSelectFromModel
from feature_selection.select_from_model import SelectFromModel

X, y = load_breast_cancer(return_X_y=True)
clf = LogisticRegression()
est1 = SelectFromModel(estimator=clf).fit(X, y)
est2 = skSelectFromModel(estimator=clf).fit(X, y)
assert np.allclose(est1.threshold_, est2.threshold_)
Xt1 = est1.transform(X)
Xt2 = est2.transform(X)
assert np.allclose(Xt1, Xt2)
print("Test passed!")