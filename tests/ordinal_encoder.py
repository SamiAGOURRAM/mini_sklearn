import numpy as np
from sklearn.preprocessing import OrdinalEncoder as skOrdinalEncoder
from preprocessing.encoding.ordinal_encoder import OrdinalEncoder

X = np.array([['cat', 'A'], ['dog', 'B'], ['cat', 'A'], ['bird', 'B']])

# Custom OrdinalEncoder
custom_encoder = OrdinalEncoder()
custom_encoded = custom_encoder.fit_transform(X)

# scikit-learn OrdinalEncoder
sk_encoder = skOrdinalEncoder()
sk_encoded = sk_encoder.fit_transform(X)

# Assertions to check if both encoders produce the same result
assert np.array_equal(custom_encoded, sk_encoded), "The custom OrdinalEncoder does not match the sklearn OrdinalEncoder."

# Output the results to verify
print("Custom OrdinalEncoder result:")
print(custom_encoded)

print("scikit-learn OrdinalEncoder result:")
print(sk_encoded)