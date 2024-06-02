import numpy as np
from sklearn.preprocessing import LabelEncoder as skLabelEncoder
from preprocessing.encoding.label_encoder import LabelEncoder

# Prepare the test data
y = np.array(['dog', 'cat', 'dog', 'fish', 'cat', 'dog', 'fish', 'dog'])

# Custom LabelEncoder
custom_encoder = LabelEncoder()
custom_encoded = custom_encoder.fit_transform(y)
custom_classes = custom_encoder.classes_

# scikit-learn LabelEncoder
sk_encoder = skLabelEncoder()
sk_encoded = sk_encoder.fit_transform(y)
sk_classes = sk_encoder.classes_

# Assertions to check if both encoders produce the same result
assert np.array_equal(custom_encoded, sk_encoded), "The custom LabelEncoder does not match the sklearn LabelEncoder."
assert np.array_equal(custom_classes, sk_classes), "The custom LabelEncoder classes do not match the sklearn LabelEncoder classes."

# Output the results to verify
print("Custom LabelEncoder encoded result:")
print(custom_encoded)

print("scikit-learn LabelEncoder encoded result:")
print(sk_encoded)

# Additional metrics to compare the encoders
from sklearn.metrics import accuracy_score

# Using the encoded labels for a simple comparison
accuracy = accuracy_score(custom_encoded, sk_encoded)
print(f"Accuracy between custom and sklearn LabelEncoder: {accuracy * 100}%")
