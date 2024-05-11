# mini_sklearn/base/base.py

class BaseEstimator:
    """Base class for all estimators in mini_sklearn."""
    
    def fit(self, X, y=None):
        """Fit estimator to data."""
        raise NotImplementedError("fit method is not implemented.")

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction."""
        raise NotImplementedError("score method is not implemented.")
    
    def check_is_fitted(self) -> bool:
        if not self._is_fitted:
            raise Exception('This estimator is not fitted.')

class TransformerMixin:
    """Mixin class for all transformers in mini_sklearn."""
    
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        raise NotImplementedError("fit_transform method is not implemented.")

    def transform(self, X):
        """Transform the data."""
        raise NotImplementedError("transform method is not implemented.")

class ClassifierMixin:
    """Mixin class for all classifiers in mini_sklearn."""
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        raise NotImplementedError("predict_proba method is not implemented.")

    def predict_log_proba(self, X):
        """Predict class log-probabilities."""
        raise NotImplementedError("predict_log_proba method is not implemented.")

class RegressorMixin:
    """Mixin class for all regressors in mini_sklearn."""
    
    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction."""
        y_pred = self.predict(X)

        # TODO: import r2_score
        return r2_score(y, y_pred, sample_weight=sample_weight)