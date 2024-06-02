import numpy as np
from sklearn.datasets import load_iris
from sklearn.impute import MissingIndicator as skMissingIndicator

from preprocessing.imputation.missing_indicator import MissingIndicator

X, _ = load_iris(return_X_y=True)
rng = np.random.RandomState(0)
missing_samples = np.arange(X.shape[0])
missing_features = rng.choice(X.shape[1], X.shape[0])
X[missing_samples, missing_features] = np.nan

est1 = MissingIndicator().fit(X)
est2 = skMissingIndicator().fit(X)
assert np.array_equal(est1.features_, est2.features_)
Xt1 = est1.transform(X)
Xt2 = est2.transform(X)
assert np.allclose(Xt1, Xt2)