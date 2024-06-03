
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import cosine_similarity as skcosine_similarity
from metrics.cosine_similarity import cosine_similarity


X, _ = load_iris(return_X_y=True)
ans1 = cosine_similarity(X)
ans2 = skcosine_similarity(X)
assert np.allclose(ans1, ans2)
ans1 = cosine_similarity(X[:100], X[100:])
ans2 = skcosine_similarity(X[:100], X[100:])
assert np.allclose(ans1, ans2)