__all__ = ['StreamClassifier']


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
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
        y_pred = np.array([i is not None for i in i_pred])
        return y_pred

    def detect(self, X):
        X = array2d(X)

        n_samples, n_features = X.shape
        if n_features > self.N_ref_:
            raise ValueError
        N_obs = self.N_obs if self.N_obs is not None else n_features

        i_pred = []
        for X_i in X:
            detection = DetectStream(X_i, N_obs,
                               self.R_pos_, self.R_neg_,
                               self.gamma, self.theta, self.D_req)
            i_pred.append(detection)
        return i_pred
