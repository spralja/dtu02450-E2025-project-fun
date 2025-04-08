from .base_model import BaseModel

import numpy as np


class LinearRegressionModel(BaseModel):
  def __init__(self, regparam=1.0, *, standardize=True):
    super().__init__(regparam, standardize=standardize)

    self._w = None

  def _preprocess(self, X, y=None):
    X, y = super()._preprocess(X, y)

    X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

    return X, y

  def fit(self, X, y):
    self._init_preprocess(X, y)

    X, y = self._preprocess(X, y)

    _, M = X.shape

    Xty = X.T @ y
    XtX = X.T @ X

    lambdaI = self.regparam * np.eye(M)

    lambdaI[0, 0] = 0

    self._w = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

  def predict(self, X):
    X, _ = self._preprocess(X)

    return X @ self._w
