# mini_sklearn/naive_bayes.py

from .base import BaseEstimator, ClassifierMixin
import numpy as np

class GaussianNB(BaseEstimator, ClassifierMixin):
    """Gaussian Naive Bayes (GaussianNB)"""
    
    def fit(self, X, y):
        """Fit Gaussian Naive Bayes model."""
        # Calculate class prior probabilities
        self.class_prior_ = np.bincount(y) / len(y)
        
        # Calculate mean and variance for each class
        self.theta_ = np.array([np.mean(X[y == c], axis=0) for c in np.unique(y)])
        self.sigma_ = np.array([np.var(X[y == c], axis=0) for c in np.unique(y)])
        
    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples."""
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) /
                                  (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)
        return np.array(joint_log_likelihood).T
    
    def predict(self, X):
        """Perform classification on samples in X."""
        joint_log_likelihood = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]
