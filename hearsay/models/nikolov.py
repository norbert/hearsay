__all__ = ['TrendClassifier',
           'TrendNormalizer',
           'BaselineNormalizer',
           'SpikeNormalizer',
           'SmoothingNormalizer',
           'LogNormalizer']


from collections import OrderedDict


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.utils import array2d, column_or_1d, check_arrays


from ..algorithms.nikolov import detect_stream
from ..normalizations import *


class TrendClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, gamma=1, theta=1, D_req=1, N_obs=None, N_ref=None):
        self.gamma = gamma
        self.theta = theta
        self.D_req = D_req
        self.N_obs = N_obs
        self.N_ref = N_ref

    def fit(self, X, y):
        X, y = check_arrays(X, y, sparse_format='dense')
        y = np.asarray(column_or_1d(y), dtype='int8')

        n_samples, n_features = X.shape
        if self.N_ref is None:
            self.N_ref = n_features
        if self.N_ref < n_features:
            X = X[:, :self.N_ref]

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
        y_pred = np.array([i is not None for i in i_pred], dtype='int8')
        return y_pred

    def detect(self, X):
        X = array2d(X)

        n_samples, n_features = X.shape
        N_obs = self.N_obs if self.N_obs is not None else n_features
        if N_obs > self.N_ref:
            raise ValueError

        i_pred = []
        for X_i in X:
            detection = detect_stream(X_i, N_obs,
                                      self.R_pos_, self.R_neg_,
                                      self.gamma, self.theta, self.D_req)
            i_pred.append(detection)
        return i_pred


class TransformerMixin(_TransformerMixin):

    def fit(self, X, y=None, **kwargs):
        return self


class TrendNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, beta=1, alpha=1.2, N_smooth=1, log=True,
                 mode='online', epsilon=0.01):
        self.beta = beta
        self.alpha = alpha
        self.N_smooth = N_smooth
        self.log = log
        self.epsilon = epsilon

        transformers = OrderedDict()
        if beta is not None:
            transformers['baseline'] = \
                BaselineNormalizer(beta, mode=mode, epsilon=epsilon)
        if alpha is not None:
            transformers['spike'] = SpikeNormalizer(alpha)
        if N_smooth is not None and N_smooth > 1:
            transformers['smoothing'] = SmoothingNormalizer(N_smooth)
        if log:
            transformers['log'] = LogNormalizer(epsilon=epsilon)
        self._transformers = transformers

    def transform(self, X, y=None):
        X = array2d(X)
        for transformer in self._transformers.values():
            X = transformer.transform(X)
        return X


class BaselineNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, beta=1, mode='online', epsilon=0.01):
        self.beta = beta
        self.mode = mode
        self.epsilon = epsilon

    def transform(self, X, y=None):
        X = array2d(X)
        return normalize_baseline(X, self.beta,
                                  mode=self.mode, epsilon=self.epsilon)


class SpikeNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, alpha=1.2):
        self.alpha = alpha

    def transform(self, X, y=None):
        X = array2d(X)
        return normalize_spikes(X, self.alpha)


class SmoothingNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, N=1):
        self.N = N

    def transform(self, X, y=None):
        X = array2d(X)
        return moving_average(X, self.N)


class LogNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon

    def transform(self, X, y=None):
        e = self.epsilon
        X = array2d(X)
        return np.log(X + e)
