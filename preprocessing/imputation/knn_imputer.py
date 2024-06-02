import numpy as np
from neighbors.nearest_neighbors import NearestNeighbors
from base.base import TransformerMixin, BaseEstimator


class KNNImputer(TransformerMixin, BaseEstimator):
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X):
        self._fit_X = X.copy()
        self.n_features_ = X.shape[1]
        self.neighbor_indices_ = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric).fit(X).kneighbors(X, return_distance=False)
        return self

    def transform(self, X):
        Xt = X.copy()
        for i in range(X.shape[0]):
            missing_features = np.isnan(Xt[i])
            for j in range(self.n_features_):
                if missing_features[j]:
                    neighbors = self.neighbor_indices_[i]
                    neighbor_values = self._fit_X[neighbors, j]
                    valid_neighbors = ~np.isnan(neighbor_values)
                    if np.any(valid_neighbors):
                        Xt[i, j] = np.mean(neighbor_values[valid_neighbors])
                    else:
                        Xt[i, j] = np.nan
        return Xt

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
