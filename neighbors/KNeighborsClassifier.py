import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
from base.base import BaseEstimator, ClassifierMixin, RegressorMixin

def _custom_cdist(self, X, Y, metric='euclidean'):
    # Initialize distance matrix
    dist_mat = np.zeros((X.shape[0], Y.shape[0]))
    
    # Calculate distances between each pair of points in X and Y using the specified metric
    if metric == 'euclidean':
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                dist_mat[i, j] = euclidean(x, y)
    elif metric == 'manhattan':
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                dist_mat[i, j] = cityblock(x, y)
    elif metric == 'cosine':
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                dist_mat[i, j] = cosine(x, y)
    else:
        raise ValueError("Unsupported distance metric: {}".format(metric))
    
    return dist_mat


class KNeighborsClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        # Store the training data
        self._fit_X = X
        
        # Store unique classes and convert labels to integers
        self.classes_, self._y = np.unique(y, return_inverse=True)
        
        return self

    def predict(self, X):
        # Calculate distances between X and training data
        dist_mat = self._custom_cdist(X, self._fit_X)
        
        # Get indices of nearest neighbors
        neigh_ind = np.argsort(dist_mat, axis=1)[:, :self.n_neighbors]
        
        # Predict labels based on majority vote
        return self.classes_[np.argmax(np.bincount(self._y[neigh_ind], axis=1))]

    def predict_proba(self, X):
        # Calculate distances between X and training data
        dist_mat = self._custom_cdist(X, self._fit_X)
        
        # Get indices of nearest neighbors
        neigh_ind = np.argsort(dist_mat, axis=1)[:, :self.n_neighbors]
        
        # Initialize probability array
        proba = np.zeros((X.shape[0], len(self.classes_)))
        
        # Count occurrences of each class among neighbors
        pred_labels = self._y[neigh_ind]
        for idx in pred_labels.T:
            proba[np.arange(X.shape[0]), idx] += 1
        
        # Normalize probabilities
        proba /= np.sum(proba, axis=1)[:, np.newaxis]
        
        return proba
