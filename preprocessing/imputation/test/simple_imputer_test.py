import numpy as np
from preprocessing.imputation.simple_imputer import SimpleImputer
from scipy.stats.mstats import mode
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer as skSimpleImputer

X, _ = load_iris(return_X_y=True)
rng = np.random.RandomState(0)
missing_samples = np.arange(X.shape[0])
missing_features = rng.choice(X.shape[1], X.shape[0])
X[missing_samples, missing_features] = np.nan

est1 = SimpleImputer(strategy="mean").fit(X)
est2 = skSimpleImputer(strategy="mean").fit(X)
assert np.allclose(est1.statistics_, est2.statistics_)
Xt1 = est1.transform(X)
Xt2 = est2.transform(X)
assert np.allclose(Xt1, Xt2)

est1 = SimpleImputer(strategy="median").fit(X)
est2 = skSimpleImputer(strategy="median").fit(X)
assert np.allclose(est1.statistics_, est2.statistics_)
Xt1 = est1.transform(X)
Xt2 = est2.transform(X)
assert np.allclose(Xt1, Xt2)

est1 = SimpleImputer(strategy="most_frequent").fit(X)
est2 = skSimpleImputer(strategy="most_frequent").fit(X)

assert np.allclose(est1.statistics_, est2.statistics_)
Xt1 = est1.transform(X)
Xt2 = est2.transform(X)
assert np.allclose(Xt1, Xt2)


est1 = SimpleImputer(strategy="constant", fill_value=0).fit(X)
est2 = skSimpleImputer(strategy="constant", fill_value=0).fit(X)
assert np.allclose(est1.statistics_, est2.statistics_)
Xt1 = est1.transform(X)
Xt2 = est2.transform(X)
assert np.allclose(Xt1, Xt2)

