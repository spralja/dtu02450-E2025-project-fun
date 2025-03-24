#%%
# from Ex2 import *


#%%
# Setup

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import pandas as pd

# Importing the data
filename = 'data/glass+identification/glass.csv'

data = pd.read_csv(filename)

attributeNames = np.asarray(data.columns)[1:]
print(attributeNames)

rawvalues = data.values
X = rawvalues[:, 1:-1]
y = rawvalues[:, -1]

N, M = X.shape

print(rawvalues.shape)
print(X.shape)


median = np.nanmedian(X, axis=0)
std = np.nanstd(X, axis=0)

X_no_nan = rawvalues[:,1:-1][~np.isnan(rawvalues).any(axis=1)]
y_no_nan = rawvalues[:,-1][~np.isnan(rawvalues).any(axis=1)]

N, M = X_no_nan.shape
print(f'X shape: {X_no_nan.shape}')

C = 7
classNames = ['building_windows_float_processed', 'building_windows_non_float_processed', 'vehicle_windows_float_processed', 'vehicle_windows_non_float_processed', 'containers', 'tableware', 'headlamps']


#%%
print(f'X shape: {X_no_nan.shape}')
print((np.ones(N) * X_no_nan.mean(axis=0).shape))
#%%

# PCA

# Subtract the mean from the data
Y = X_no_nan - np.ones([N, 1]) * X_no_nan.mean(axis=0)

# Divide by standard deviation
Y = Y / (np.ones([N, 1]) * np.std(Y, axis=0))

# Multiplying the RI by 1000
# Y[:,0] = Y[:,0] * 100



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
Z = Y @ V


print(f'Z shape: {Z.shape}')
print(f'V shape: {V.shape}')
print(f'Y shape: {Y.shape}')
print(f'y shape: {y_no_nan.shape}')
#%%


# Indices of the principal components to be plotted
i = 0
j = 1


# Plot PCA of the data
f = plt.figure()
plt.title("PCA", fontsize=18)
# Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y_no_nan == c
    plt.plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5)
plt.legend(classNames)
plt.xlabel("PC{0}".format(i + 1), fontsize=14)
plt.ylabel("PC{0}".format(j + 1), fontsize=14)

# Output result to screen
plt.show()


#%%
pcs = [0, 1, 2]
legendStrs = ["PC" + str(e + 1) for e in pcs]
bw = 0.2 #Bar width
print(M)
r = np.arange(1, M + 1)

for i in pcs:
    plt.bar(r + i * bw, V[:, i], width=bw)

plt.xticks(r + bw, attributeNames[:-1],rotation=45)
plt.xlabel("Attributes")
plt.ylabel("Component coefficients")
plt.legend(legendStrs)
plt.grid()
plt.title("PCA Component Coefficients")
plt.show()


#%%
print(attributeNames)