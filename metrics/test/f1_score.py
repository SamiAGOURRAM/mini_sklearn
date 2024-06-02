import numpy as np
from sklearn.metrics import f1_score as skf1_score
from metrics.classification import f1_score


# binary
for i in range(10):
    rng = np.random.RandomState(i)
    y_true = rng.randint(2, size=10)
    y_pred = rng.randint(2, size=10)
    score1 = f1_score(y_true, y_pred, average="binary")
    score2 = skf1_score(y_true, y_pred, average="binary")
    assert np.isclose(score1, score2)


# multiclass
for i in range(10):
    for average in (None, "micro", "macro", "weighted"):
        rng = np.random.RandomState(i)
        y_true = rng.randint(3, size=10)
        y_pred = rng.randint(3, size=10)
        score1 = f1_score(y_true, y_pred, average=average)
        score2 = skf1_score(y_true, y_pred, average=average)
        if average is None:
            assert np.array_equal(score1, score2)
        else:
            assert np.isclose(score1, score2)