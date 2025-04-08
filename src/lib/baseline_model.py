from .base_model import BaseModel

import numpy as np


class BaselineModel(BaseModel):
  def __init__(self, regparam=1.0, *, standardize=True):
    super().__init__(regparam, standardize=standardize)

    self._maj = None

  def fit(self, X, y):
    values, counts = np.unique(y, return_counts=True)
    self._maj = values[np.argmax(counts)]
    return self

  def predict(self, X):
    return np.full(shape=(X.shape[0],), fill_value=self._maj)
