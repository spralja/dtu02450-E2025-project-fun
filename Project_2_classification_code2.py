#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm
from sklearn import tree

from dtuimldmtools import rlr_validate

#%% Load the dataset
# Importing the data
filename = 'data/glass+identification/glass.csv'

data = pd.read_csv(filename)


attributeNames = np.asarray(data.columns)[1:]

rawvalues = data.values
X = rawvalues[:, 1:-1] # Exclude first (Id) and last (Type) columns
y = rawvalues[:, -1] # Last column (Type)

N, M = X.shape

print(f'Number of observations: {N}')

C = 7
classNames = ['building_windows_float_processed', 'building_windows_non_float_processed', 'vehicle_windows_float_processed', 'vehicle_windows_non_float_processed', 'containers', 'tableware', 'headlamps']

from lib import LogisticRegressionModel, BaselineModel


# %%
model = LogisticRegressionModel(2.0)
model.fit(X, y)
from sklearn.metrics import accuracy_score

print("Training accuracy: ", accuracy_score(y, model.predict(X)))

# %%

model = BaselineModel()
model.fit(X, y)

print("Training accuracy: ", accuracy_score(y, model.predict(X)))
# %%
models = [BaselineModel, LogisticRegressionModel]
Kin, Kout = 3, 8
RNDSTATE = 395168

from sklearn import model_selection

outer_cv = model_selection.KFold(n_splits=Kin, shuffle=True, random_state=RNDSTATE)
inner_cv = model_selection.KFold(n_splits=Kout, shuffle=True, random_state=RNDSTATE)

outer_cv_split = list(outer_cv.split(X, y))

regparamss = [[1.0], np.logspace(-4, 4, num=400)]

def evaluate(Model, X_train, y_train, X_test, y_test, *, regparam=1.0):
  model = Model(regparam=regparam)
  model.fit(X_train, y_train)

  acc = accuracy_score(y_test, model.predict(X_test))

  return acc


def evaluate_fold(Model, X, y, *, regparam=1.0):
  accs = []
  for train_idx, test_idx in inner_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    acc = evaluate(Model, X_train, y_train, X_test, y_test, regparam=regparam)

    accs.append(acc)

  return sum(accs) / len(accs)


for Model, regparams in zip(models, regparamss):
  for train_idx, test_idx in outer_cv_split:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    accs = []
    for regparam in regparams:
      acc = evaluate_fold(Model, X_train, y_train, regparam=regparam)

      accs.append(acc)

    regparam = regparams[np.argmax(accs[-1])]

    print(max(accs))

    acc = evaluate(Model, X_train, y_train, X_test, y_test, regparam=regparam)

    print("ACC =", acc)


# %%
print(np.array(accs))

print(regparams[np.argmax(accs)])
# %%
