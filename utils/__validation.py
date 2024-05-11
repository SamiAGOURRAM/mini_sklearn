# mini_sklearn/utils/validation.py

import numpy as np

def _check_sample_weight(sample_weight, X):
    """Validate sample weights.

    Parameters
    ----------
    sample_weight : array-like of shape (n_samples,)
        Sample weights.

    X : array-like of shape (n_samples, n_features)
        Training data.

    Returns
    -------
    sample_weight : ndarray of shape (n_samples,)
        Validated sample weights.
    """
    if sample_weight is None:
        return np.ones(X.shape[0], dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight)
        if np.any(sample_weight < 0):
            raise ValueError("Sample weights must be non-negative.")
        if len(sample_weight) != X.shape[0]:
            raise ValueError("Sample weights length must be equal to the number of samples.")
        return sample_weight




# mini_sklearn/utils/validation.py

def _check_partial_fit_first_call(estimator, classes):
    """Check if partial_fit is called for the first time.

    Parameters
    ----------
    estimator : object
        Estimator instance.

    classes : array-like of shape (n_classes,)
        Unique class labels.

    Returns
    -------
    bool
        True if partial_fit is called for the first time, False otherwise.
    """
    if getattr(estimator, "classes_", None) is None and classes is None:
        raise ValueError(
            "classes must be passed on the first call to partial_fit."
        )
    return getattr(estimator, "classes_", None) is None


