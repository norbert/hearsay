__all__ = ['StreamClassifier',
           'BaselineSpikeNormalizer',
           'SmoothingNormalizer',
           'LogNormalizer']


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils import array2d, column_or_1d, check_arrays


from .algorithm import DetectStream


class StreamClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, N_obs=None, gamma=1, theta=1, D_req=1):
        self.N_obs = N_obs
        self.gamma = gamma
        self.theta = theta
        self.D_req = D_req

    def fit(self, X, y):
        X, y = check_arrays(X, y, sparse_format='dense')
        y = column_or_1d(y)
        y = np.asarray(y, dtype=np.int8)

        n_samples, n_features = X.shape
        self.N_ref_ = n_features

        unique_y = np.unique(y)
        if not np.array_equal(unique_y, (0, 1)):
            raise ValueError

        R_pos = X[(y == 1)]
        R_neg = X[(y == 0)]
        self.R_pos_ = R_pos
        self.R_neg_ = R_neg

        return self

    def predict(self, X):
        i_pred = self.detect(X)
        y_pred = np.array([i is not None for i in i_pred], dtype=np.int8)
        return y_pred

    def detect(self, X):
        X = array2d(X)

        n_samples, n_features = X.shape
        if n_features > self.N_ref_:
            raise ValueError
        N_obs = self.N_obs if self.N_obs is not None else n_features

        i_pred = []
        for X_i in X:
            detection = DetectStream(iter(X_i), N_obs,
                               self.R_pos_, self.R_neg_,
                               self.gamma, self.theta, self.D_req)
            i_pred.append(detection)
        return i_pred


class BaselineSpikeNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, alpha = 1, beta = 1, epsilon = 0.01):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        X = array2d(X)
        e = self.epsilon

        def normalize_baseline(signal, beta):
            norm = np.sum(signal, 1)[:,None]
            values = ((signal + e) / (norm + e)) ** beta
            return values
        def normalize_spikes(signal, alpha):
            values = np.abs(signal[:,1:] - signal[:,0:-1]) ** alpha
            return values

        values = X
        if not self.beta is None:
            values = normalize_baseline(values, self.beta)
        if not self.alpha is None:
            values = normalize_spikes(values, self.alpha)

        return values


class SmoothingNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, n = 1, epsilon = 0.01):
        self.n = n
        self.epsilon = epsilon

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        e = self.epsilon
        X = array2d(X)

        # http://stackoverflow.com/a/14314054
        def moving_average(X, n):
            ret = np.cumsum(X, 1) + e
            ret[:,n:] = ret[:,n:] - ret[:,:-n]
            return ret[:,(n - 1):] / n

        return moving_average(X, self.n)


class LogNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, epsilon = 0.01):
        self.epsilon = epsilon

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        e = self.epsilon
        X = array2d(X)
        return np.log(X + e)