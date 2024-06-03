from preprocessing.transformers.scaler import StandardScaler
import numpy as np

# Create a StandardScaler instance
scaler = StandardScaler()

# Test data
data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])

# Fit the scaler and check attributes
assert np.allclose(scaler.fit(data).mean_, [0.5, 0.5])
assert np.allclose(scaler.scale_, [0.5, 0.5])

# Transform the data
transformed_data = scaler.transform(data)

# Expected transformed data
expected_data = np.array([[-1., -1.],
                            [-1., -1.],
                            [ 1.,  1.],
                            [ 1.,  1.]])

# Check if transformed data matches expected
assert np.allclose(transformed_data, expected_data)

print("All tests passed!")