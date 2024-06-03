import numpy as np
from sklearn.preprocessing import OneHotEncoder as skOneHotEncoder

from preprocessing.encoding.one_hot_encoder import OneHotEncoder

X = np.array([['cat', 'A'], ['dog', 'B'], ['cat', 'A'], ['bird', 'B']])

# Custom OneHotEncoder
custom_encoder = OneHotEncoder()
custom_encoded = custom_encoder.fit_transform(X)

# scikit-learn OneHotEncoder
sk_encoder = skOneHotEncoder(sparse=False)
sk_encoded = sk_encoder.fit_transform(X)

# Assertions to check if both encoders produce the same result
assert np.array_equal(custom_encoded, sk_encoded), "The custom OneHotEncoder does not match the sklearn OneHotEncoder."

# Output the results to verify
print("Custom OneHotEncoder result:")
print(custom_encoded)

print("scikit-learn OneHotEncoder result:")
print(sk_encoded)