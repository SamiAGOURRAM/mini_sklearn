import numpy as np
from sklearn.impute import KNNImputer as skKNNImputer
from preprocessing.imputation.knn_imputer import KNNImputer


# Generate some data with missing values
X = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])

# Impute missing values using your KNNImputer
your_imputer = KNNImputer(n_neighbors=2)
X_imputed_yours = your_imputer.fit_transform(X)

# Impute missing values using scikit-learn's KNNImputer
sk_imputer = skKNNImputer(n_neighbors=2)
X_imputed_sklearn = sk_imputer.fit_transform(X)

print(X_imputed_yours, X_imputed_sklearn)

# Check if the imputed values are the same
assert np.allclose(X_imputed_yours, X_imputed_sklearn)

print("Test passed successfully.")
