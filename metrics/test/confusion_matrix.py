import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import confusion_matrix as skconfusion_matrix
from metrics.classification import confusion_matrix

# binary
for i in range(10):
    rng = np.random.RandomState(i)
    y_true = rng.randint(2, size=10)
    y_pred = rng.randint(2, size=10)
    score1 = confusion_matrix(y_true, y_pred)
    score2 = skconfusion_matrix(y_true, y_pred)
    assert np.array_equal(score1, score2)


# multiclass
for i in range(10):
    rng = np.random.RandomState(i)
    y_true = rng.randint(3, size=10)
    y_pred = rng.randint(3, size=10)
    score1 = confusion_matrix(y_true, y_pred)
    score2 = skconfusion_matrix(y_true, y_pred)
    assert np.array_equal(score1, score2)