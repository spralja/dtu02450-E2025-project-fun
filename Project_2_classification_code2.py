#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm
from sklearn import tree

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
Kin, Kout = 5, 5
RNDSTATE = 395168

from sklearn import model_selection

outer_cv = model_selection.KFold(n_splits=Kin, shuffle=True, random_state=RNDSTATE)
inner_cv = model_selection.KFold(n_splits=Kout, shuffle=True, random_state=RNDSTATE)

outer_cv_split = list(outer_cv.split(X, y))

accs = []
regparams = [0.1, 0.2, 0.5, 1, 1.5]

for m_index, Model in enumerate(models):
  accs.append([])
  for (train_idx_outer, test_idx_outer), regparam in zip(outer_cv_split, regparams):
    accs[-1].append([])

    X_train_outer, X_test_outer = X[train_idx_outer], X[test_idx_outer]
    y_train_outer, y_test_outer = y[train_idx_outer], y[test_idx_outer]


    for train_idx, test_idx in inner_cv.split(X_train_outer, y_train_outer):
      X_train, X_test = X_train_outer[train_idx], X_train_outer[test_idx]
      y_train, y_test = y_train_outer[train_idx], y_train_outer[test_idx]

      model = Model(regparam=regparam)
      model.fit(X_train, y_train)

      acc = accuracy_score(y_test, model.predict(X_test))

      accs[-1][-1].append(acc)

    accs[-1][-1] = sum(accs[-1][-1]) / len(accs[-1][-1])
    
  
# %%
print(np.array(accs))
# %%
