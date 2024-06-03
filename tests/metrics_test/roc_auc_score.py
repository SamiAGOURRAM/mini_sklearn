from sklearn.metrics import roc_auc_score as skroc_auc_score
from metrics.roc_auc_score import roc_auc_score
import numpy as np

for i in range(10):
    rng = np.random.RandomState(i)
    y_true = rng.randint(2, size=10)
    y_score = rng.randint(11, size=10) / 10
    ans1 = roc_auc_score(y_true, y_score)
    ans2 = skroc_auc_score(y_true, y_score)
    assert np.allclose(ans1, ans2)