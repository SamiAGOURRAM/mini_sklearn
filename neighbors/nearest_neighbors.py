from scipy.spatial.distance import cdist
from base.base import TransformerMixin, BaseEstimator
import numpy as np

class NearestNeighbors(TransformerMixin, BaseEstimator):
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X):
        self._fit_X = X
        self._is_fitted = True
        return self

    def kneighbors(self, X, n_neighbors=1, return_distance=True):
        """
        Finds the K-neighbors of a point.

        Parameters:
        - X (array-like): The query point or points.
        - n_neighbors (int): The number of neighbors to find.
        - return_distance (bool): If True, returns the distances to the neighbors along with the neighbors.

        Returns:
        - neigh_dist (array): Array representing the lengths to points, only present if return_distance=True.
        - neigh_ind (array): Indices of the nearest points in the population matrix.
        """
        neigh_ind = np.argsort(cdist(X, self._fit_X, metric=self.metric), axis=1)[:, :n_neighbors]
        if return_distance:
            neigh_dist = np.zeros(neigh_ind.shape)
            for i, indices in enumerate(neigh_ind):
                neigh_dist[i, :] = cdist([X[i]], self._fit_X[indices], metric=self.metric)
            return neigh_dist, neigh_ind
        return neigh_ind


    def transform(self, X):
        distances, indices = self.kneighbors(X)
        return indices

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
