import numpy as np
from sklearn.metrics import mean_squared_error as skmean_squared_error
from sklearn.metrics import mean_absolute_error as skmean_absolute_error
from sklearn.metrics import r2_score as skr2_score

from metrics.regression import mean_squared_error, r2_score, mean_absolute_error


#MSE
for i in range(10):
    rng = np.random.RandomState(i)
    y_true = rng.rand(10)
    y_pred = rng.rand(10)
    score1 = mean_squared_error(y_true, y_pred)
    score2 = skmean_squared_error(y_true, y_pred)
    assert np.isclose(score1, score2)

#MAE
for i in range(10):
    rng = np.random.RandomState(i)
    y_true = rng.rand(10)
    y_pred = rng.rand(10)
    score1 = mean_absolute_error(y_true, y_pred)
    score2 = skmean_absolute_error(y_true, y_pred)
    assert np.isclose(score1, score2)

#R2 score
for i in range(10):
    rng = np.random.RandomState(i)
    y_true = rng.rand(10)
    y_pred = rng.rand(10)
    score1 = r2_score(y_true, y_pred)
    score2 = skr2_score(y_true, y_pred)
    assert np.isclose(score1, score2)

