import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import load_breast_cancer
from linear_model.logistic_regression import LogisticRegression
from sklearn.feature_selection import RFE as skRFE
from feature_selection.RFE import RFE

X, y = load_breast_cancer(return_X_y=True)

clf = LogisticRegression()
est1 = RFE(estimator=clf).fit(X, y)
est2 = skRFE(estimator=clf).fit(X, y)
assert np.array_equal(est1.support_, est2.support_)
assert np.array_equal(est1.ranking_, est2.ranking_)
Xt1 = est1.transform(X)
Xt2 = est2.transform(X)
assert np.allclose(Xt1, Xt2)

print("Test passed!")