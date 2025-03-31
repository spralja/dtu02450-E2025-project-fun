from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
  def __init__(self, regparam=1.0, *, standardize=True):
    self.standardize = standardize
    self.regparam = regparam

    self.X_mean = None
    self.X_std = None

  def _init_preprocess(self, X, y=None):
    if self.standardize:
      self.X_mean = np.mean(X, axis=0)
      self.X_std = np.std(X, axis=0)

  def _preprocess(self, X, y=None):
    if self.standardize:
      return (X - self.X_mean) / self.X_std, y
    else:
      return X, y

  @abstractmethod
  def fit(self, X, y=None):
    pass

  @abstractmethod
  def predict(self, X):
    pass
