import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import cosine_distances as skcosine_distances

from metrics.cosine_distances import cosine_distances

X, _ = load_iris(return_X_y=True)
ans1 = cosine_distances(X)
ans2 = skcosine_distances(X)
assert np.allclose(ans1, ans2)
ans1 = cosine_distances(X[:100], X[100:])
ans2 = skcosine_distances(X[:100], X[100:])
assert np.allclose(ans1, ans2)