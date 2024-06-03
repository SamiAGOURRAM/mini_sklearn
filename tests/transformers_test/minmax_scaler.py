import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler as skMinMaxScaler
from preprocessing.transformers.scaler import MinMaxScaler

X, _ = load_iris(return_X_y=True)
sc1 = MinMaxScaler().fit(X)
sc2 = skMinMaxScaler().fit(X)
assert np.allclose(sc1.data_min_, sc2.data_min_)
assert np.allclose(sc1.data_max_, sc2.data_max_)
assert np.allclose(sc1.data_range_, sc2.data_range_)
assert np.allclose(sc1.scale_, sc2.scale_)
Xt1 = sc1.transform(X)
Xt2 = sc2.transform(X)
assert np.allclose(Xt1, Xt2)
print("test passed!")