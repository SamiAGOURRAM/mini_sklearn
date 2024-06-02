import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split
from model_selection.__split import train_test_split
from numpy.testing import assert_array_equal

# Generate synthetic data
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# Split using your function
X_train_custom, X_test_custom, y_train_custom, y_test_custom = train_test_split(X, y, test_size=0.3, random_state=42)

# Split using scikit-learn
X_train_sk, X_test_sk, y_train_sk, y_test_sk = sk_train_test_split(X, y, test_size=0.3, random_state=42)

# Compare the outputs
assert_array_equal(y_test_custom, y_test_sk)



print("Arrays are equal")


