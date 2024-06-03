import numpy as np
from sklearn.preprocessing import Normalizer as SklearnNormalizer
from preprocessing.transformers.normalizer import Normalizer


X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Initialize the Normalizer
normalizer = Normalizer(norm='l2')

# Fit and transform the data
X_normalized = normalizer.fit_transform(X)

# Compare with sklearn's Normalizer
sklearn_normalizer = SklearnNormalizer(norm='l2')
X_normalized_sklearn = sklearn_normalizer.fit_transform(X)

# Assert that the results are the same
assert np.allclose(X_normalized, X_normalized_sklearn)
print("test passed!")