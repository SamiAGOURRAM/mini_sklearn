import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold as skVarianceThreshold
from feature_selection.variance_threshold import VarianceThreshold

X, _ = load_iris(return_X_y=True)

X, _ = load_iris(return_X_y=True)
X[:, [0, 2]] = 0
est1 = VarianceThreshold().fit(X)
est2 = skVarianceThreshold().fit(X)
assert np.allclose(est1.variances_, est2.variances_)
Xt1 = est1.transform(X)
Xt2 = est2.transform(X)
assert np.allclose(Xt1, Xt2)
print("Test passed!")