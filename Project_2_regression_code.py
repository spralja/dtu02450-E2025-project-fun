#%%
%pip install scikit-learn

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm

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

mean = np.mean(X, 0)
std = np.std(X, 0)

# Making a standardized version of the data
X_hat = (X - np.ones((N, M)) * mean) / (np.ones((N, M)) * std)

#%% Fit a linear regression model to the data

# print(attributeNames)

RI_vals = np.array(X_hat[:, 0]).reshape(-1, 1)
Ca_vals = np.array(X_hat[:, 6]).reshape(-1, 1)
print(RI_vals.shape)
print(Ca_vals.shape)

#%%
# Copied from exercise 5.2.2


# Fit ordinary least squares regression model
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(Ca_vals, RI_vals)
# Compute model output:
y_est = model.predict(Ca_vals)
# Or equivalently:
# y_est = model.intercept_ + X @ model.coef_


# Plot original data and the model output
f = plt.figure()

plt.plot(Ca_vals, RI_vals, ".")
# plt.plot(RI_vals, y_true, "-")
plt.plot(Ca_vals, y_est, "-")
plt.xlabel("Ca_vals")
plt.ylabel("RI_vals")
plt.legend(["Training data", "Data generator", "Regression fit (model)"])

plt.show()

print("Ran Exercise 5.2.2")
