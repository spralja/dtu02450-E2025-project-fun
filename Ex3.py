#%%
from Ex2 import *


#%%
# Setup

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd

# Importing the data
filename = 'data/water_potability.csv'

data = pd.read_csv(filename)

attributeNames = np.asarray(data.columns)

rawvalues = data.values
X = rawvalues[:, :-1]
y = rawvalues[:, -1]

N, M = X.shape



median = np.nanmedian(X, axis=0)
std = np.nanstd(X, axis=0)

X_no_nan = rawvalues[:,:-1][~np.isnan(rawvalues).any(axis=1)]
y_no_nan = rawvalues[:,-1][~np.isnan(rawvalues).any(axis=1)]

C = 2
classNames = ["Not potable", "Potable"]

#%%

# PCA

# Subtract the mean from the data
Y = X_no_nan - np.ones((len(X_no_nan[:,1]), 1)) * X_no_nan.mean(axis=0)

# PCA by computing SVD of Y
U, S, Vh = svd(Y, full_matrices=False)
V = Vh.T

rho = (S*S) / (S*S).sum()

threshold = 0.9


#%%
# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()



#%%
# Project the centered data onto principal component space
# Note: Make absolutely sure you understand what the @ symbol 
# does by inspecing the numpy documentation!
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = plt.figure()
plt.title("NanoNose data: PCA")
# Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5)
plt.legend(classNames)
plt.xlabel("PC{0}".format(i + 1))
plt.ylabel("PC{0}".format(j + 1))

# Output result to screen
plt.show()