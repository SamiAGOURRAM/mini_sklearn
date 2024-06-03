
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold as skKFold
from model_selection.__split import KFold

X, y = load_diabetes(return_X_y=True)
cv1 = KFold(n_splits=5)
cv2 = skKFold(n_splits=5)
for (train1, test1), (train2, test2) in zip(cv1.split(X, y), cv2.split(X, y)):
    assert np.array_equal(train1, train2)
    assert np.array_equal(test1, test2)


X, y = load_diabetes(return_X_y=True)
cv1 = KFold(n_splits=5, shuffle=True, random_state=0)
cv2 = skKFold(n_splits=5, shuffle=True, random_state=0)
for (train1, test1), (train2, test2) in zip(cv1.split(X, y), cv2.split(X, y)):
    assert np.array_equal(train1, train2)
    assert np.array_equal(test1, test2)
