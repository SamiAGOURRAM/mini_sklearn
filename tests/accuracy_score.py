from sklearn.metrics import accuracy_score as skaccuracy_score
import numpy as np
from metrics.classification import accuracy_score
# binary
for i in range(10):
    rng = np.random.RandomState(i)
    y_true = rng.randint(2, size=10)
    y_pred = rng.randint(2, size=10)
    score1 = accuracy_score(y_true, y_pred)
    score2 = skaccuracy_score(y_true, y_pred)
    assert np.isclose(score1, score2)

# multiclass
for i in range(10):
    rng = np.random.RandomState(i)
    y_true = rng.randint(3, size=10)
    y_pred = rng.randint(3, size=10)
    score1 = accuracy_score(y_true, y_pred)
    score2 = skaccuracy_score(y_true, y_pred)
    assert np.isclose(score1, score2)