import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import euclidean_distances as skeuclidean_distances
from metrics.euclidian_distances import euclidean_distances

X, _ = load_iris(return_X_y=True)
ans1 = euclidean_distances(X)
ans2 = skeuclidean_distances(X)
assert np.allclose(ans1, ans2, atol=1e-6)
ans1 = euclidean_distances(X[:100], X[100:])
ans2 = skeuclidean_distances(X[:100], X[100:], squared=False)
assert np.allclose(ans1, ans2)