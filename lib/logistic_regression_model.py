from .base_model import BaseModel

import sklearn.linear_model as lm


class LogisticRegressionModel(BaseModel):
  def __init__(self, regparam=1.0, *, standardize=True):
    super().__init__(regparam, standardize=standardize)

    self._model = lm.LogisticRegression(solver='lbfgs', C=1 / self.regparam)

  def fit(self, X, y):
    self._init_preprocess(X, y)

    X, y = self._preprocess(X, y)

    self._model.fit(X, y)

    return self

  def predict(self, X):
    X, _ = self._preprocess(X)
    return self._model.predict(X)
