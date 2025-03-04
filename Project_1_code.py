#%%
# Download the data from our Github repository
import os
import requests

data_url = 'https://raw.githubusercontent.com/spralja/dtu02450-E2025-project-fun/88525af3cc564763aedbe8c78e07aa4f81b44e12/data/glass%2Bidentification/glass.csv'

# Download the data (Written by copilot)
r = requests.get(data_url)
os.makedirs('data/glass+identification', exist_ok=True)
with open('data/glass+identification/glass.csv', 'wb') as f:
    f.write(r.content)

#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd

#%%
# Importing the data
filename = 'data/glass+identification/glass.csv'

data = pd.read_csv(filename)
#%%

attributeNames = np.asarray(data.columns)[1:]

rawvalues = data.values
X = rawvalues[:, 1:-1] # Exclude first (Id) and last (Type) columns
y = rawvalues[:, -1] # Last column (Type)

N, M = X.shape

if X.any() < 0:
    print('Negative values in X')
else:
    print('No negative values in X')
if X.all() >= 0:
    print('All values in X are positive')
if 1 <= X[:,0].all() <= 4:
    print('The RI values are reasonable')
else:
    print('The RI values are not reasonable')
    print(max(X[:,0]))
    print(min(X[:,0]))

# Compute statistics
median_values = np.nanmedian(X, axis=0)
std_values = np.nanstd(X, axis=0, ddof=1)
mean_values = np.nanmean(X, axis=0)
range_values = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)

# Print calculated values
print("Feature Names:\n", data.columns.values)
print("\nMean of the features:\n", mean_values)
print("\nMedian of the features:\n", median_values)
print("\nStandard deviation of the features:\n", std_values)
print("\nRange of the features:\n", range_values)

# Check for missing values (NaN)
nan_count = np.sum(np.isnan(X), axis=0)
print("\nNaN count per feature:", nan_count)
print("\nTotal NaN values in Xset:", np.sum(np.isnan(X)))
print("\nRows with at least one NaN:", np.sum(data.isna().any(axis=1)))

#%%
nan_count = np.sum(np.isnan(X), axis=0)
print(f'Nan count: {nan_count}')
print(f'Nan percentage: {nan_count/len(X[:,0])*100}%')
nan_total = np.sum(np.isnan(X))
print(f'Nan')

# %%

# Generate x-axis labels (indices)
x_labels = attributeNames[1:-1]

# Plot the bar graph
plt.bar(x_labels, X[69][1:], color='blue')

# Label axes
plt.xlabel('Oxide')
plt.ylabel('% content')
plt.title('Example Datapoint')

# Show the plot
plt.show()

#%%
sum_temp = np.sum(X, axis=1)
nan_row_sum = np.sum(np.isnan(sum_temp))
print(f'Nan row sum: {nan_row_sum}')
print(f'Nan row percentage: {round(nan_row_sum/len(X[:,0])*100,2)}%')

#%%
#Dataset without NaN values
is_not_nan = np.isnan(X) < 1
# X_no_nan = X[~np.isnan(X).any(axis=1)]
X_no_nan = X[is_not_nan.all(axis=1)]


#%%
# Copied from excercise 2.3.2
plt.figure(figsize=(8, 7))
plt.suptitle("Histograms of attributes", fontsize=18)
u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)
for i in range(M):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(X[:, i], color=(0.2, 0.8 - i * 0.2*(4/M), 0.4),bins=20)
    plt.xlabel(attributeNames[i], fontsize=16)
    plt.ylim(0, N / 2)
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
plt.show()

#%%

plt.figure()
plt.boxplot(X_no_nan)
plt.xticks(range(1, 10), attributeNames[:-1], rotation=45)
plt.title("Boxplot of data without NaN values")
plt.show()


#%%

# Looking at a standardized version of the data through boxplots

X_hat = [(X[:, i][~np.isnan(X[:, i])] - median[i])/ std[i] for i in range(9)]

plt.figure()
plt.boxplot(X_hat)
plt.xticks(range(1, 10), attributeNames[:-1], rotation=45, fontsize=14)
plt.title("Boxplot of standardized data", fontsize=18)
plt.show()


#%%
# Similarities

Cor_mat = np.empty((M,M))
for i in range(M):
    for j in range(M):
        Cor_mat[i,j] = np.corrcoef(X_no_nan[:,i], X_no_nan[:,j])[0,1]
Cor_sign = np.sign(Cor_mat)
plt.figure()
plt.title("Correlation matrix", fontsize=18)
plt.pcolormesh(np.abs(Cor_mat))
plt.xticks(range(0,9), attributeNames[:-1], fontsize=14)
plt.yticks(range(0,9), attributeNames[:-1], fontsize=14)
plt.colorbar()
plt.show()
print(Cor_mat[1,4])
print(Cor_mat[2,4])
print(f'Highest correlation: {np.max(np.abs(Cor_mat[Cor_mat < 0.9]))}')


#%%
C = 7
classNames = ['building_windows_float_processed', 'building_windows_non_float_processed', 'vehicle_windows_float_processed', 'vehicle_windows_non_float_processed', 'containers', 'tableware', 'headlamps']

plt.figure(figsize=(12, 10))
plt.suptitle("Scatter plots of attributes comparissons", fontsize=24)
plt.tight_layout()
for m1 in range(M):
    for m2 in range(M):
        plt.subplot(M, M, m1 * M + m2 + 1)
        for c in range(C):
            class_mask = y == c
            plt.plot(np.array(X[class_mask, m2]), np.array(X[class_mask, m1]), ".")
            if m1 == M - 1:
                plt.xlabel(attributeNames[m2], fontsize=16)
            else:
                plt.xticks([])
            if m2 == 0:
                plt.ylabel(attributeNames[m1], fontsize=16)
            else:
                plt.yticks([])
                            
# The legend blocks the view of some of the plots, 
# and doesn't add much value for the context of this plot
# plt.legend(classNames)

plt.show()



#%%
# Setup for 2nd part (PCA)

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
