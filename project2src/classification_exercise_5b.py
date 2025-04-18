#%% Import Modules
from lib import LinearRegressionModel
import pandas as pd
import numpy as np

#%% Load the dataset
# Importing the data
filename = 'data/glass+identification/glass.csv'

data = pd.read_csv(filename)


attributeNames = np.asarray(data.columns)[2:-1]

rawvalues = data.values
X = rawvalues[:, 2:-1] # Exclude first (Id) and last (Type) columns, and also RI
y = rawvalues[:, 1] # RI column

N, M = X.shape

print(f'Number of observations: {N}')
print(M)

C = 7
classNames = ['building_windows_float_processed', 'building_windows_non_float_processed', 'vehicle_windows_float_processed', #'vehicle_windows_non_float_processed', 
              'containers', 'tableware', 'headlamps']


#%% Set CONSTANTS
regparam = 10e-7

#%%
model = LinearRegressionModel(regparam=regparam)

model.fit(X, y)

#%%

coef = model._w[1:]

print(coef)

np.abs(coef).mean(axis=0)
# %%
avgw = np.abs(coef).mean(axis=0)
classLabels = attributeNames

print(len(coef))
print(len(attributeNames));print(attributeNames)

# %%
import matplotlib.pyplot as plt
### Generated by ChatGPT (OpenAI) ###
avg_abs_weights = np.mean(np.abs(coef), axis=0)

# Plot as bar chart
plt.figure(figsize=(10, 6))
plt.bar(attributeNames, coef)
plt.title('Average Absolute Feature Weights Across All Classes')
plt.xlabel('Features')
plt.ylabel('Average |Weight| (log)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
### ----------------------------- ###
# %%


np.square(y - model.predict(X)).sum(axis=0) / y.shape[0]
# %%
