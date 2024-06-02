import numpy as np

def euclidean_distances(X, Y=None, *, squared=False):
    """Compute Euclidean distances between samples in X and Y.

    Read more in the :ref:`User Guide <euclidean_distances>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        First set of samples.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), \
            default=None
        Second set of samples. If `None`, the distances are computed between
        all pairs of samples in `X`.

    squared : bool, default=False
        Return squared Euclidean distances.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y) or \
                (n_samples_X, n_samples_X)
        Array containing the pairwise Euclidean distances. If `squared=True`,
        the array contains the squared distances.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = [[0, 0], [1, 1]]
    >>> euclidean_distances(X, X)
    array([[0.        , 1.41421356],
           [1.41421356, 0.        ]])

    >>> euclidean_distances(X, X, squared=True)
    array([[0., 2.],
           [2., 0.]])
    """
    # Check if Y is provided
    if Y is None:
        Y = X

    # Compute the pairwise Euclidean distances
    distances = -2 * np.dot(X, Y.T) + np.sum(X ** 2, axis=1)[:, np.newaxis] + np.sum(Y ** 2, axis=1)

    if not squared:
        distances = np.sqrt(distances)

    return distances
